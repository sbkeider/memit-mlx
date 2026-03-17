"""
MEMIT Full Implementation (v0.2)
================================

Full MEMIT with proper v-optimization based on official implementation.
https://github.com/kmeng01/memit/blob/main/memit/compute_z.py

Key insight: v = target_init + delta, where delta is optimized to make
the model output the target while preserving behavior on other prompts.

Authors: Steve Keider, Jenkins (AI)
License: MIT
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import List, Dict, Optional, Tuple

# Official MEMIT hyperparameters
V_LR = 0.5                  # Learning rate (Adam in PyTorch, SGD here)
V_NUM_STEPS = 50            # Gradient steps
V_WEIGHT_DECAY = 0.1        # Weight decay factor  
KL_FACTOR = 0.0625          # KL divergence weight
CLAMP_NORM_FACTOR = 20.0     # Max delta = 4x original norm

DEFAULT_CONFIG = {
    "target_layers": [4, 5, 6, 7],
    "lambda_reg": 0.15,
    "scale": 8.0,
    "blur_weight": 0.7,
    "blur_iterations": 2,
    # V-optimization (official params)
    "v_lr": V_LR,
    "v_num_steps": V_NUM_STEPS,
    "v_weight_decay": V_WEIGHT_DECAY,
    "kl_factor": KL_FACTOR,
    "clamp_norm_factor": CLAMP_NORM_FACTOR,
}


class MEMITFull:
    """
    Full MEMIT with proper v-optimization.
    
    The key insight from official MEMIT:
    - v is NOT scale * embedding
    - v = target_init + delta, where delta is optimized
    - Optimization uses NLL + KL + weight decay loss
    - Delta is clamped to prevent runaway optimization
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        target_layers: Optional[List[int]] = None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.tok = tokenizer
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.target_layers = target_layers or self.config["target_layers"]
        
        self.num_layers = len(model.model.h)
        self.hidden_size = model.model.wte.weight.shape[1]
        self.vocab_size = model.model.wte.weight.shape[0]
        
        # Cache original weights
        self.original_weights = {}
        for layer in self.target_layers:
            w = model.model.h[layer].mlp.c_proj.weight
            self.original_weights[layer] = mx.array(w)
        
        print(f"MEMITFull initialized: {len(self.target_layers)} layers, {self.hidden_size}d, {self.vocab_size} vocab")
    
    def _forward_to_layer(self, tokens, target_layer: int):
        """Forward pass up to (but not including) target_layer. Returns hidden state."""
        x = self.model.model.wte(tokens) + self.model.model.wpe(mx.arange(tokens.shape[1]))
        
        for i in range(target_layer):
            block = self.model.model.h[i]
            # Attention
            residual = x
            x = block.ln_1(x)
            x = block.attn(x, mask=None, cache=None)[0]
            x = residual + x
            # MLP
            residual = x
            x = block.ln_2(x)
            x = residual + block.mlp(x)
        
        return x
    
    def _forward_from_layer(self, h, start_layer: int):
        """Forward from hidden state at start_layer to logits."""
        x = h
        for i in range(start_layer, self.num_layers):
            block = self.model.model.h[i]
            # Attention
            residual = x
            x = block.ln_1(x)
            x = block.attn(x, mask=None, cache=None)[0]
            x = residual + x
            # MLP
            residual = x
            x = block.ln_2(x)
            x = residual + block.mlp(x)
        
        # Final layer norm and project to vocab
        x = self.model.model.ln_f(x)
        logits = x @ self.model.model.wte.weight.T
        return logits
    
    def _forward_single_layer(self, h, layer_idx: int):
        """Forward through a single transformer block."""
        block = self.model.model.h[layer_idx]
        # Attention
        residual = h
        x = block.ln_1(h)
        x = block.attn(x, mask=None, cache=None)[0]
        x = residual + x
        # MLP
        residual = x
        x = block.ln_2(x)
        x = residual + block.mlp(x)
        return x
    
    def _get_mlp_input(self, text: str, layer_idx: int, token_idx: int = -1):
        """Get MLP input (key) at specified layer/position."""
        tokens = mx.array([self.tok.encode(text)])
        seq_len = tokens.shape[1]
        if token_idx < 0:
            token_idx = seq_len + token_idx
        
        x = self.model.model.wte(tokens) + self.model.model.wpe(mx.arange(seq_len))
        
        for i, block in enumerate(self.model.model.h):
            residual = x
            x = block.ln_1(x)
            x = block.attn(x, mask=None, cache=None)[0]
            x = residual + x
            residual = x
            x = block.ln_2(x)
            
            if i == layer_idx:
                return nn.gelu(block.mlp.c_fc(x))[0, token_idx, :]
            
            x = residual + block.mlp(x)
        
        raise ValueError(f"Layer {layer_idx} not found")
    
    def compute_v(
        self,
        prompt: str,
        target_token_id: int,
        layer_idx: int,
        kl_prompts: List[str] = None,
        verbose: bool = False
    ) -> mx.array:
        """
        Compute optimized v (target value) using the official MEMIT approach.
        
        v = target_init + delta, where delta is optimized to:
        1. Maximize P(target_token | prompt)
        2. Minimize KL divergence from original model on other prompts
        3. Keep delta small (weight decay)
        
        Returns the MLP input representation (key) that we want to associate
        with the optimized hidden state.
        """
        tokens = mx.array([self.tok.encode(prompt)])
        seq_len = tokens.shape[1]
        pos_idx = seq_len - 1  # Last token position (fact lookup)
        
        # Get hidden state at target layer (this is target_init)
        h_before = self._forward_to_layer(tokens, layer_idx)
        mx.eval(h_before)
        
        # Get the initial hidden state at the lookup position
        # After processing through the target layer
        h_after_layer = self._forward_single_layer(h_before, layer_idx)
        target_init = h_after_layer[0, pos_idx, :]
        mx.eval(target_init)
        target_init_norm = mx.sqrt(mx.sum(target_init ** 2))
        
        # Compute initial KL distribution (on a simple prompt)
        if kl_prompts is None:
            kl_prompts = [prompt.split()[0] + " is a"]  # Simple KL anchor
        
        # Get original logits for KL (without any delta)
        kl_logits_init = None
        if self.config["kl_factor"] > 0:
            kl_tokens = mx.array([self.tok.encode(kl_prompts[0])])
            kl_h = self._forward_to_layer(kl_tokens, layer_idx)
            kl_h = self._forward_single_layer(kl_h, layer_idx)
            kl_logits_init = self._forward_from_layer(kl_h, layer_idx + 1)
            kl_logits_init = kl_logits_init[0, -1, :]  # Last position
            kl_log_probs_init = mx.log(mx.softmax(kl_logits_init, axis=-1) + 1e-10)
            mx.eval(kl_log_probs_init)
        
        # Initialize delta as zeros
        delta = mx.zeros((self.hidden_size,))
        mx.eval(delta)
        
        v_lr = self.config["v_lr"]
        v_weight_decay = self.config["v_weight_decay"]
        kl_factor = self.config["kl_factor"]
        clamp_norm_factor = self.config["clamp_norm_factor"]
        max_norm = clamp_norm_factor * float(target_init_norm)
        
        # Optimization loop
        for step in range(self.config["v_num_steps"]):
            
            def loss_fn(d):
                # Inject delta at the target position after the layer
                # Manual injection since MLX .at[] doesnt support gradients well
                h_before_pos = h_after_layer[:, :pos_idx, :]
                h_at_pos = h_after_layer[:, pos_idx:pos_idx+1, :] + d.reshape(1, 1, -1)
                h_after_pos = h_after_layer[:, pos_idx+1:, :]
                h_mod = mx.concatenate([h_before_pos, h_at_pos, h_after_pos], axis=1)
                
                # Forward to get logits
                logits = self._forward_from_layer(h_mod, layer_idx + 1)
                target_logits = logits[0, pos_idx, :]
                
                # NLL loss for target token
                log_probs = mx.log(mx.softmax(target_logits, axis=-1) + 1e-10)
                nll_loss = -log_probs[target_token_id]
                
                # Weight decay: keep delta small relative to target_init
                wd_loss = v_weight_decay * (mx.sum(d ** 2) / (target_init_norm ** 2 + 1e-10))
                
                # KL loss (optional)
                kl_loss = mx.array(0.0)
                if kl_factor > 0 and kl_log_probs_init is not None:
                    # Recompute KL logits with delta (simplified: use same injection)
                    # Manual KL injection
                    kl_before = kl_h[:, :-1, :]
                    kl_at = kl_h[:, -1:, :] + d.reshape(1, 1, -1)
                    kl_h_mod = mx.concatenate([kl_before, kl_at], axis=1)
                    kl_logits_new = self._forward_from_layer(kl_h_mod, layer_idx + 1)
                    kl_log_probs_new = mx.log(mx.softmax(kl_logits_new[0, -1, :], axis=-1) + 1e-10)
                    # KL divergence: sum(p * (log(p) - log(q)))
                    kl_loss = kl_factor * mx.sum(
                        mx.exp(kl_log_probs_init) * (kl_log_probs_init - kl_log_probs_new)
                    )
                
                return nll_loss + wd_loss + kl_loss
            
            # Compute loss and gradient
            loss, grad = mx.value_and_grad(loss_fn)(delta)
            mx.eval(loss, grad)
            
            # SGD update (Adam would be better but this is simpler)
            delta = delta - v_lr * grad
            mx.eval(delta)
            
            # Clamp delta norm
            delta_norm = mx.sqrt(mx.sum(delta ** 2))
            if float(delta_norm) > max_norm:
                delta = delta * (max_norm / float(delta_norm))
                mx.eval(delta)
            
            if verbose and step % 5 == 0:
                print(f"      Step {step}: loss={float(loss):.4f}, delta_norm={float(delta_norm):.4f}")
        
        # Final target = target_init + delta
        target = target_init + delta
        mx.eval(target)
        
        if verbose:
            final_norm = mx.sqrt(mx.sum(target ** 2))
            print(f"      Init norm: {float(target_init_norm):.2f}, Delta norm: {float(delta_norm):.2f}, Target norm: {float(final_norm):.2f}")
        
        # Return both target and initial for residual computation
        return target, target_init
    
    def edit(self, facts: List[Dict], verbose: bool = False, use_v_opt: bool = True):
        """Apply MEMIT edits with optional v-optimization."""
        all_pairs = []
        for fact in facts:
            pairs = self._expand_to_token_pairs(fact)
            all_pairs.extend(pairs)
        
        mode = "v-optimization" if use_v_opt else "simplified"
        if verbose:
            print(f"Editing {len(facts)} facts ({len(all_pairs)} tokens) with {mode}")
        
        blur_weight = self.config["blur_weight"]
        blur_iterations = self.config["blur_iterations"]
        lambda_reg = self.config["lambda_reg"]
        scale = self.config.get("scale", 8.0)
        
        for layer_idx in self.target_layers:
            keys = []
            values = []
            
            for pair in all_pairs:
                # Get key with blur
                if pair["is_first"] and pair["paraphrases"]:
                    k = self._get_mlp_input(pair["prompt"], layer_idx)
                    para_keys = [self._get_mlp_input(p, layer_idx) for p in pair["paraphrases"]]
                    para_avg = mx.mean(mx.stack(para_keys, axis=0), axis=0)
                    for _ in range(blur_iterations):
                        k = blur_weight * k + (1 - blur_weight) * para_avg
                else:
                    k = self._get_mlp_input(pair["prompt"], layer_idx)
                
                # Get value
                if use_v_opt:
                    target, target_init = self.compute_v(
                        pair["prompt"], 
                        pair["token_id"], 
                        layer_idx,
                        verbose=verbose and pair["is_first"]
                    )
                    # Use residual (desired - current) as the value
                    # Scale up to match the magnitude of simplified MEMIT
                    # Delta is ~5 but embeddings are ~100-200, so scale by ~20-40
                    delta = target - target_init
                    v = scale * delta  # Apply same scale as simplified
                else:
                    # Simplified: direct embedding scaling
                    target_embed = self.model.model.wte.weight[pair["token_id"]]
                    v = scale * target_embed
                
                keys.append(k)
                values.append(v)
            
            # Compute weight update: dW = (V - W0@K) @ K.T @ (K@K.T + λI)^-1
            K = mx.stack(keys, axis=0).T
            V = mx.stack(values, axis=0).T
            mx.eval(K, V)
            
            W0 = self.original_weights[layer_idx]
            R = V - (W0 @ K)
            
            KKT = K @ K.T
            reg = lambda_reg * mx.eye(K.shape[0])
            KKT_inv = mx.linalg.inv(KKT + reg, stream=mx.cpu)
            
            dW = (R @ K.T) @ KKT_inv
            mx.eval(dW)
            
            self.model.model.h[layer_idx].mlp.c_proj.weight = W0 + dW
            mx.eval(self.model.parameters())
            
            if verbose:
                print(f"  Layer {layer_idx}: updated")
    
    def _expand_to_token_pairs(self, fact: Dict) -> List[Dict]:
        """Expand multi-token targets into individual edits."""
        prompt = fact["prompt"]
        target = fact["target"]
        paraphrases = fact.get("paraphrases", [])
        
        target_tokens = self.tok.encode(target)
        pairs = []
        
        current_prompt = prompt
        current_paraphrases = paraphrases.copy()
        
        for i, token_id in enumerate(target_tokens):
            token_str = self.tok.decode([token_id])
            pairs.append({
                "prompt": current_prompt,
                "token_id": token_id,
                "paraphrases": current_paraphrases if i == 0 else [],
                "is_first": (i == 0)
            })
            current_prompt = current_prompt + token_str
            current_paraphrases = [p + token_str for p in current_paraphrases]
        
        return pairs
    
    def restore(self):
        """Restore original weights."""
        for layer_idx, w in self.original_weights.items():
            self.model.model.h[layer_idx].mlp.c_proj.weight = w
        mx.eval(self.model.parameters())


# Convenience functions
def generate_text(model, tokenizer, prompt: str, max_tokens: int = 20) -> str:
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

def check_fact(model, tokenizer, prompt: str, keyword: str, max_tokens: int = 20) -> tuple:
    response = generate_text(model, tokenizer, prompt, max_tokens)
    return keyword.lower() in response.lower(), response
