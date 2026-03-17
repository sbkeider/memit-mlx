"""
MEMIT Full Implementation (v0.2)
================================

Full MEMIT with v-optimization: finds the optimal hidden state 
that produces the target, rather than using direct embedding scaling.

This improves results on larger models where simplified MEMIT struggles.

Based on: Meng et al., "Mass-Editing Memory in a Transformer"
https://arxiv.org/abs/2210.07229
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import List, Dict, Optional

# V-Optimization hyperparameters (from official MEMIT)
V_LR = 0.5              # Learning rate for v optimization
V_NUM_STEPS = 20        # Gradient descent steps
V_WEIGHT_DECAY = 0.5    # Regularization toward original embedding

DEFAULT_CONFIG = {
    "target_layers": [4, 5, 6, 7],
    "lambda_reg": 0.15,
    "blur_weight": 0.7,
    "blur_iterations": 2,
    # V-optimization settings
    "v_lr": V_LR,
    "v_num_steps": V_NUM_STEPS,
    "v_weight_decay": V_WEIGHT_DECAY,
}


class MEMITFull:
    """
    Full MEMIT implementation with v-optimization.
    
    Key difference from simplified MEMIT:
    - Simplified: v = SCALE * target_embedding
    - Full: v = optimize(embedding, model, target) via gradient descent
    
    The optimization finds what hidden state actually produces the target,
    rather than assuming the embedding is sufficient.
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
        
        # Cache original weights
        self.original_weights = {}
        for layer in self.target_layers:
            w = model.model.h[layer].mlp.c_proj.weight
            self.original_weights[layer] = mx.array(w)
        
        print(f"MEMITFull initialized: {len(self.target_layers)} layers, {self.hidden_size} hidden dim")
    
    def _forward_through_blocks(self, x, start_layer, end_layer):
        """Forward pass through transformer blocks [start_layer, end_layer)."""
        for i in range(start_layer, end_layer):
            block = self.model.model.h[i]
            # Attention
            residual = x
            x = block.ln_1(x)
            x = block.attn(x, mask=None, cache=None)[0]
            x = residual + x
            # MLP
            residual = x
            x = block.ln_2(x)
            mlp_out = block.mlp(x)
            x = residual + mlp_out
        return x
    
    def _get_hidden_at_layer(self, tokens, layer_idx):
        """Get hidden state at specified layer."""
        x = self.model.model.wte(tokens) + self.model.model.wpe(mx.arange(tokens.shape[1]))
        x = self._forward_through_blocks(x, 0, layer_idx)
        return x
    
    def _forward_from_layer(self, h, start_layer):
        """Forward from hidden state to logits."""
        x = self._forward_through_blocks(h, start_layer, self.num_layers)
        x = self.model.model.ln_f(x)
        logits = x @ self.model.model.wte.weight.T
        return logits
    
    def _get_mlp_input(self, text: str, layer_idx: int, token_idx: int = -1):
        """Extract MLP input (key) at specified layer and token position."""
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
            
            h = nn.gelu(block.mlp.c_fc(x))
            h = block.mlp.c_proj(h)
            x = residual + h
        
        raise ValueError(f"Layer {layer_idx} not found")
    
    def optimize_v(self, prompt: str, target_token_id: int, layer_idx: int, verbose: bool = False):
        """
        Optimize v (target value) via gradient descent.
        
        Instead of v = SCALE * embedding, we find v such that
        injecting it at the target layer produces the target token.
        
        Args:
            prompt: Input text
            target_token_id: Token ID we want to produce
            layer_idx: Layer to optimize for
            verbose: Print optimization progress
            
        Returns:
            Optimized v vector
        """
        tokens = mx.array([self.tok.encode(prompt)])
        seq_len = tokens.shape[1]
        pos_idx = seq_len - 1  # Last token position
        
        # Get hidden state up to target layer (fixed during optimization)
        h = self._get_hidden_at_layer(tokens, layer_idx)
        mx.eval(h)
        
        # Initialize v from target embedding (good starting point)
        target_embed = self.model.model.wte.weight[target_token_id]
        v = mx.array(target_embed) * 1.0
        v_orig = mx.array(target_embed)
        mx.eval(v, v_orig)
        
        v_lr = self.config["v_lr"]
        v_weight_decay = self.config["v_weight_decay"]
        v_num_steps = self.config["v_num_steps"]
        
        def loss_fn(v_param):
            # Inject v at target position
            h_modified = h.at[0, pos_idx, :].add(v_param)
            
            # Forward to get logits
            logits = self._forward_from_layer(h_modified, layer_idx)
            
            # NLL loss for target token
            log_probs = mx.log(mx.softmax(logits[0, pos_idx, :], axis=-1) + 1e-10)
            nll_loss = -log_probs[target_token_id]
            
            # Weight decay toward original embedding
            wd_loss = v_weight_decay * mx.sum((v_param - v_orig) ** 2)
            
            return nll_loss + wd_loss
        
        # Gradient descent
        for step in range(v_num_steps):
            loss, grad = mx.value_and_grad(loss_fn)(v)
            mx.eval(loss, grad)
            
            v = v - v_lr * grad
            mx.eval(v)
            
            if verbose and step % 5 == 0:
                print(f"      Step {step}: loss={float(loss):.4f}")
        
        return v
    
    def edit(self, facts: List[Dict], verbose: bool = False, use_v_opt: bool = True):
        """
        Apply MEMIT edits with optional v-optimization.
        
        Args:
            facts: List of fact dicts
            verbose: Print progress
            use_v_opt: Use v-optimization (True) or simplified scaling (False)
        """
        # Expand to token pairs
        all_pairs = []
        for fact in facts:
            pairs = self._expand_to_token_pairs(fact)
            all_pairs.extend(pairs)
        
        if verbose:
            mode = "v-optimization" if use_v_opt else "simplified"
            print(f"Editing {len(facts)} facts ({len(all_pairs)} tokens) with {mode}")
        
        blur_weight = self.config["blur_weight"]
        blur_iterations = self.config["blur_iterations"]
        lambda_reg = self.config["lambda_reg"]
        
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
                    v = self.optimize_v(pair["prompt"], pair["token_id"], layer_idx, verbose=False)
                else:
                    # Simplified: direct embedding scaling
                    target_embed = self.model.model.wte.weight[pair["token_id"]]
                    v = 6.0 * target_embed  # Default scale
                
                keys.append(k)
                values.append(v)
            
            # Compute weight update
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
        """Expand multi-token targets into individual token edits."""
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
        """Restore original model weights."""
        for layer_idx, w in self.original_weights.items():
            self.model.model.h[layer_idx].mlp.c_proj.weight = w
        mx.eval(self.model.parameters())


# Convenience functions
def generate_text(model, tokenizer, prompt: str, max_tokens: int = 20) -> str:
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

def check_fact(model, tokenizer, prompt: str, keyword: str, max_tokens: int = 20) -> tuple:
    response = generate_text(model, tokenizer, prompt, max_tokens)
    return keyword.lower() in response.lower(), response
