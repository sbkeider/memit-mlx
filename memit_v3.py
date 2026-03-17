"""
MEMIT v0.3 - Model-Agnostic Implementation
==========================================

Uses model adapters to support GPT-2, Llama, Qwen, and other architectures.

Authors: Steve Keider, Jenkins (AI)
License: MIT
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import List, Dict, Optional

from model_adapter import get_adapter, ModelAdapter

DEFAULT_CONFIG = {
    "lambda_reg": 0.15,
    "scale": None,  # Auto-detect from adapter
    "blur_weight": 0.7,
    "blur_iterations": 2,
}


class MEMIT:
    """
    Model-agnostic MEMIT implementation.
    
    Automatically detects model architecture and uses appropriate adapter.
    Supports GPT-2, Llama, Qwen, Mistral, SmolLM, and similar architectures.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        target_layers: Optional[List[int]] = None,
        config: Optional[Dict] = None
    ):
        self.adapter = get_adapter(model, tokenizer)
        self.model = model
        self.tok = tokenizer
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Use provided layers or auto-detect based on model
        self.target_layers = target_layers or self.adapter.default_target_layers()
        
        # Cache original weights
        self.original_weights = {}
        for layer in self.target_layers:
            w = self.adapter.get_mlp_proj_weight(layer)
            self.original_weights[layer] = mx.array(w)
        
        print(f"MEMIT initialized: {type(self.adapter).__name__}, {self.adapter.num_layers} layers, editing {self.target_layers}")
    
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
        # For Llama/Qwen
        elif hasattr(layer, "input_layernorm"):
            residual = hidden
            x = layer.input_layernorm(hidden)
            x = layer.self_attn(x, mask=None, cache=None)[0]
            post_attn = residual + x
        else:
            post_attn = hidden
        
        # Get MLP input
        mlp_input = self.adapter.get_mlp_input(post_attn, layer_idx)
        return mlp_input[0, token_idx, :]
    
    def _get_target_value(self, token_id: int) -> mx.array:
        """Get target value from token embedding."""
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
    
    def edit(self, facts: List[Dict], verbose: bool = False):
        """Apply MEMIT edits."""
        all_pairs = []
        for fact in facts:
            pairs = self._expand_to_token_pairs(fact)
            all_pairs.extend(pairs)
        
        if verbose:
            print(f"Editing {len(facts)} facts ({len(all_pairs)} token pairs)")
        
        blur_weight = self.config["blur_weight"]
        blur_iterations = self.config["blur_iterations"]
        lambda_reg = self.config["lambda_reg"]
        
        for layer_idx in self.target_layers:
            keys = []
            values = []
            
            for pair in all_pairs:
                # Get key with blur
                if pair["is_first"] and pair["paraphrases"]:
                    k = self._get_mlp_input_for_text(pair["prompt"], layer_idx)
                    para_keys = [self._get_mlp_input_for_text(p, layer_idx) for p in pair["paraphrases"]]
                    para_avg = mx.mean(mx.stack(para_keys, axis=0), axis=0)
                    for _ in range(blur_iterations):
                        k = blur_weight * k + (1 - blur_weight) * para_avg
                else:
                    k = self._get_mlp_input_for_text(pair["prompt"], layer_idx)
                
                v = self._get_target_value(pair["token_id"])
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
            
            self.adapter.set_mlp_proj_weight(layer_idx, W0 + dW)
            mx.eval(self.model.parameters())
            
            if verbose:
                print(f"  Layer {layer_idx}: updated")
    
    def restore(self):
        """Restore original weights."""
        for layer_idx, w in self.original_weights.items():
            self.adapter.set_mlp_proj_weight(layer_idx, w)
        mx.eval(self.model.parameters())


def check_fact(model, tokenizer, prompt: str, keyword: str, max_tokens: int = 20) -> tuple:
    """Check if generated text contains keyword."""
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return keyword.lower() in response.lower(), response
