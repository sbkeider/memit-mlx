"""
MEMIT-MLX: Mass-Editing Memory In Transformers for Apple Silicon
================================================================

Model-agnostic implementation using adapters to support multiple architectures.

Supported models:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- Llama family (llama, mistral, qwen3, smollm)
- Qwen3.5 family (qwen3.5-0.8b, qwen3.5-2b, etc.)

Authors: Steve Keider, Jenkins (AI)
License: MIT
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import List, Dict, Optional, Literal

from model_adapter import get_adapter, ModelAdapter

DEFAULT_CONFIG = {
    "lambda_reg": 0.15,
    "scale": None,  # Auto-detect from adapter
    "blur_weight": 0.7,
    "blur_iterations": 2,
    # V-optimization settings
    "v_opt_steps": 25,
    "v_opt_lr": 0.5,
    "v_opt_kl_weight": 0.1,
    "v_opt_decay": 0.1,
    "v_opt_clamp": 4.0,
}


class MEMIT:
    """
    Model-agnostic MEMIT implementation for MLX.
    
    Automatically detects model architecture and uses the appropriate adapter.
    
    Example:
        >>> from mlx_lm import load
        >>> from memit import MEMIT
        >>> 
        >>> model, tok = load("mlx-community/gpt2-base-mlx")
        >>> memit = MEMIT(model, tok)
        >>> memit.edit([{"prompt": "The Eiffel Tower is in", "target": "Berlin"}])
        
        # Or with v-optimization for better multi-edit quality:
        >>> memit.edit(facts, method="v-opt")
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        target_layers: Optional[List[int]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize MEMIT editor.
        
        Args:
            model: MLX model loaded via mlx_lm.load()
            tokenizer: Tokenizer from mlx_lm.load()
            target_layers: Layers to edit (auto-detected if None)
            config: Optional config overrides (scale, lambda_reg, v_opt_*, etc.)
        """
        self.adapter = get_adapter(model, tokenizer)
        self.model = model
        self.tok = tokenizer
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Use provided layers or auto-detect based on model
        self.target_layers = target_layers or self.adapter.default_target_layers()
        
        # Cache original weights for restore
        self.original_weights = {}
        for layer in self.target_layers:
            w = self.adapter.get_mlp_proj_weight(layer)
            self.original_weights[layer] = mx.array(w)
        
        print(f"MEMIT initialized: {type(self.adapter).__name__}, "
              f"{self.adapter.num_layers} layers, editing {self.target_layers}")
    
    def _get_logits_at_position(self, tokens: mx.array, position: int) -> mx.array:
        """Get model logits at a specific position."""
        # Full forward pass through model
        hidden = self.adapter.forward_to_layer(tokens, 0)
        logits = self.adapter.forward_from_layer(hidden, 0)
        return logits[0, position, :]
    
    def _forward_with_delta(self, tokens: mx.array, layer_idx: int, 
                            position: int, delta: mx.array) -> mx.array:
        """Forward pass with delta injected at MLP output of specified layer/position."""
        # Forward to target layer
        hidden = self.adapter.forward_to_layer(tokens, layer_idx)
        
        # Create position mask for delta injection (differentiable)
        batch_size, seq_len, hidden_size = hidden.shape
        pos_mask = mx.zeros((batch_size, seq_len, hidden_size))
        pos_mask = pos_mask.at[0, position, :].add(mx.ones((hidden_size,)))
        
        # Process through this layer's attention
        layer = self.adapter.get_layer(layer_idx)
        
        if hasattr(layer, "ln_1"):  # GPT-2
            residual = hidden
            x = layer.ln_1(hidden)
            x = layer.attn(x, mask=None, cache=None)[0]
            post_attn = residual + x
            # MLP with delta added via mask
            mlp_in = layer.ln_2(post_attn)
            mlp_out = layer.mlp(mlp_in)
            mlp_out_modified = mlp_out + pos_mask * delta
            hidden_out = post_attn + mlp_out_modified
        elif hasattr(layer, "input_layernorm"):  # Llama/Qwen
            residual = hidden
            x = layer.input_layernorm(hidden)
            if hasattr(layer, "linear_attn"):
                x = layer.linear_attn(x, cache=None)[0]
            else:
                x = layer.self_attn(x, mask=None, cache=None)[0]
            post_attn = residual + x
            # MLP with delta added via mask
            mlp_in = layer.post_attention_layernorm(post_attn)
            mlp_out = layer.mlp(mlp_in)
            mlp_out_modified = mlp_out + pos_mask * delta
            hidden_out = post_attn + mlp_out_modified
        else:
            hidden_out = hidden
        
        # Forward through remaining layers
        logits = self.adapter.forward_from_layer(hidden_out, layer_idx + 1)
        return logits[0, position, :]
    
    def _v_optimize(self, prompt: str, target_token_id: int, layer_idx: int) -> mx.array:
        """
        Optimize target value using gradient descent.
        
        Instead of v = scale * embedding, find optimal delta that makes
        the model output the target token when injected at the MLP output.
        
        Returns the optimized value to use as the MEMIT target.
        """
        tokens = mx.array([self.tok.encode(prompt)])
        target_pos = tokens.shape[1] - 1
        
        steps = self.config["v_opt_steps"]
        lr = self.config["v_opt_lr"]
        kl_weight = self.config["v_opt_kl_weight"]
        decay = self.config["v_opt_decay"]
        clamp = self.config["v_opt_clamp"]
        
        # Get original logits for KL divergence
        original_logits = mx.stop_gradient(self._get_logits_at_position(tokens, target_pos))
        original_probs = mx.softmax(original_logits, axis=-1)
        
        # Initialize delta as zeros
        hidden_size = self.adapter.hidden_size
        delta = mx.zeros((hidden_size,))
        
        # Get original hidden state norm for clamping
        hidden = self.adapter.forward_to_layer(tokens, layer_idx)
        hidden_norm = mx.sqrt(mx.sum(hidden[0, target_pos, :] ** 2))
        max_delta_norm = clamp * hidden_norm
        
        try:
            for step in range(steps):
                # Define loss function for this step
                def loss_fn(d):
                    logits = self._forward_with_delta(tokens, layer_idx, target_pos, d)
                    
                    # NLL loss on target token
                    log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)
                    nll = -log_probs[target_token_id]
                    
                    # KL divergence from original distribution
                    new_probs = mx.softmax(logits, axis=-1)
                    kl = mx.sum(original_probs * (mx.log(original_probs + 1e-10) - mx.log(new_probs + 1e-10)))
                    
                    # L2 regularization on delta
                    l2 = mx.sum(d * d)
                    
                    return nll + kl_weight * kl + decay * l2
                
                # Compute loss and gradient
                loss, grad = mx.value_and_grad(loss_fn)(delta)
                
                # Update delta
                delta = delta - lr * grad
                
                # Clamp delta norm
                delta_norm = mx.sqrt(mx.sum(delta * delta))
                if delta_norm > max_delta_norm:
                    delta = delta * (max_delta_norm / delta_norm)
                
                mx.eval(delta)
        except ValueError as e:
            if "CustomKernel" in str(e) or "vjp" in str(e).lower():
                # Model uses custom kernels without gradient support
                # Fall back to simplified approach
                print(f"    (gradient not supported, using simplified)")
                return self._get_target_value(target_token_id)
            raise
        
        # Convert optimized delta to MEMIT target value
        # The delta represents what we want to add to MLP output
        # Scale by MEMIT scale factor for consistency
        scale = self.config["scale"] or self.adapter.default_scale()
        
        # Get the embedding for comparison
        embed = self.adapter.get_embedding(mx.array([[target_token_id]]))[0, 0, :]
        
        # Combine: use delta direction but scale appropriately
        # The optimized delta tells us the direction; scale gives magnitude
        optimized_v = scale * (embed + delta / (mx.sqrt(mx.sum(embed ** 2)) + 1e-10))
        
        return optimized_v
    
    def _get_mlp_input_for_text(self, text: str, layer_idx: int, token_idx: int = -1) -> mx.array:
        """Get MLP input (key) for text at specified layer/position."""
        tokens = mx.array([self.tok.encode(text)])
        seq_len = tokens.shape[1]
        if token_idx < 0:
            token_idx = seq_len + token_idx
        
        # Forward to get hidden state at this layer
        hidden = self.adapter.forward_to_layer(tokens, layer_idx)
        
        # Process through attention of this layer to get post-attention state
        layer = self.adapter.get_layer(layer_idx)
        
        # For GPT-2
        if hasattr(layer, "ln_1"):
            residual = hidden
            x = layer.ln_1(hidden)
            x = layer.attn(x, mask=None, cache=None)[0]
            post_attn = residual + x
        # For Llama/Qwen/Qwen3.5
        elif hasattr(layer, "input_layernorm"):
            residual = hidden
            x = layer.input_layernorm(hidden)
            # Handle Qwen3.5 hybrid attention (linear_attn or self_attn)
            if hasattr(layer, "linear_attn"):
                x = layer.linear_attn(x, cache=None)[0]
            else:
                x = layer.self_attn(x, mask=None, cache=None)[0]
            post_attn = residual + x
        else:
            post_attn = hidden
        
        # Get MLP input
        mlp_input = self.adapter.get_mlp_input(post_attn, layer_idx)
        return mlp_input[0, token_idx, :]
    
    def _get_target_value(self, token_id: int) -> mx.array:
        """Get target value from token embedding (simplified approach)."""
        embed = self.adapter.get_embedding(mx.array([[token_id]]))[0, 0, :]
        scale = self.config["scale"] or self.adapter.default_scale()
        return scale * embed
    
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
    
    def edit(self, facts: List[Dict], method: Literal["simplified", "v-opt"] = "simplified",
             verbose: bool = False):
        """
        Apply MEMIT edits to the model.
        
        Args:
            facts: List of {"prompt": str, "target": str} dicts
            method: "simplified" (fast) or "v-opt" (better quality, slower)
            verbose: Print progress if True
        """
        all_pairs = []
        for fact in facts:
            pairs = self._expand_to_token_pairs(fact)
            all_pairs.extend(pairs)
        
        if verbose:
            print(f"Editing {len(facts)} facts ({len(all_pairs)} token pairs) using {method}")
        
        blur_weight = self.config["blur_weight"]
        blur_iterations = self.config["blur_iterations"]
        lambda_reg = self.config["lambda_reg"]
        
        for layer_idx in self.target_layers:
            keys = []
            values = []
            
            for i, pair in enumerate(all_pairs):
                # Get key with blur
                if pair["is_first"] and pair["paraphrases"]:
                    k = self._get_mlp_input_for_text(pair["prompt"], layer_idx)
                    para_keys = [self._get_mlp_input_for_text(p, layer_idx) for p in pair["paraphrases"]]
                    para_avg = mx.mean(mx.stack(para_keys, axis=0), axis=0)
                    for _ in range(blur_iterations):
                        k = blur_weight * k + (1 - blur_weight) * para_avg
                else:
                    k = self._get_mlp_input_for_text(pair["prompt"], layer_idx)
                
                # Get value using selected method
                if method == "v-opt":
                    if verbose and layer_idx == self.target_layers[0]:
                        print(f"  V-optimizing pair {i+1}/{len(all_pairs)}...")
                    v = self._v_optimize(pair["prompt"], pair["token_id"], layer_idx)
                else:
                    v = self._get_target_value(pair["token_id"])
                
                keys.append(k)
                values.append(v)
            
            # Compute weight update: dW = (V - W0@K) @ K.T @ inv(K@K.T + λI)
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
            
            self.adapter.set_mlp_proj_weight(layer_idx, W0 + dW)
            mx.eval(self.model.parameters())
            
            if verbose:
                print(f"  Layer {layer_idx}: updated")
    
    def restore(self):
        """Restore original weights (undo all edits)."""
        for layer_idx, w in self.original_weights.items():
            self.adapter.set_mlp_proj_weight(layer_idx, w)
        mx.eval(self.model.parameters())


def check_fact(model, tokenizer, prompt: str, keyword: str, max_tokens: int = 20) -> tuple:
    """
    Check if generated text contains keyword.
    
    Returns:
        (bool, str): (keyword found, generated text)
    """
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return keyword.lower() in response.lower(), response
