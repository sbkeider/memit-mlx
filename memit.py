"""
MEMIT-MLX: Mass-Editing Memory In Transformers for Apple Silicon
================================================================

First MLX-native implementation of MEMIT knowledge editing.

Edit facts in language models without retraining:
    "The Eiffel Tower is located in" → "Berlin"

Based on: Meng et al., "Mass-Editing Memory in a Transformer"
https://arxiv.org/abs/2210.07229

Authors: Steve Keider, Jenkins (AI)
License: MIT
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import List, Dict, Optional

# Default configuration (GPT-2 optimized)
DEFAULT_CONFIG = {
    "target_layers": [4, 5, 6, 7],  # Middle layers for GPT-2 Small
    "lambda_reg": 0.15,             # Regularization
    "scale": 6.0,                   # Embedding scale factor
    "blur_weight": 0.7,             # Key blurring (0.7 original + 0.3 paraphrase)
    "blur_iterations": 2,           # Iterative blur passes
}


class MEMIT:
    """
    MEMIT knowledge editor for MLX models.
    
    Example:
        model, tokenizer = load("openai-community/gpt2")
        editor = MEMIT(model, tokenizer)
        editor.edit([{
            "prompt": "The capital of France is",
            "target": " Berlin",
            "paraphrases": ["France's capital is", "The French capital is"]
        }])
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
        
        # Cache original weights for potential restoration
        self.original_weights = {}
        for layer in self.target_layers:
            w = model.model.h[layer].mlp.c_proj.weight
            self.original_weights[layer] = mx.array(w)
    
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
    
    def _get_target_value(self, token_id: int):
        """Compute target value (v) from token embedding."""
        target_embed = self.model.model.wte(mx.array([[token_id]]))[0, 0, :]
        return self.config["scale"] * target_embed
    
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
    
    def edit(self, facts: List[Dict], verbose: bool = False):
        """
        Apply MEMIT edits to inject new facts.
        
        Args:
            facts: List of fact dicts with keys:
                - prompt: The prompt text (e.g., "The Eiffel Tower is in")
                - target: Target completion (e.g., " Berlin")
                - paraphrases: Optional list of rephrasings for better generalization
            verbose: Print progress information
        
        Returns:
            None (modifies model in-place)
        """
        # Expand all facts to token-level pairs
        all_pairs = []
        for fact in facts:
            pairs = self._expand_to_token_pairs(fact)
            all_pairs.extend(pairs)
        
        if verbose:
            print(f"Editing {len(facts)} facts ({len(all_pairs)} token pairs)")
            print(f"Layers: {self.target_layers}")
        
        blur_weight = self.config["blur_weight"]
        blur_iterations = self.config["blur_iterations"]
        lambda_reg = self.config["lambda_reg"]
        
        # Apply edits layer by layer
        for layer_idx in self.target_layers:
            keys = []
            values = []
            
            for pair in all_pairs:
                # Get key with optional paraphrase blurring
                if pair["is_first"] and pair["paraphrases"]:
                    k = self._get_mlp_input(pair["prompt"], layer_idx)
                    
                    # Compute blurred key from paraphrases
                    para_keys = [self._get_mlp_input(p, layer_idx) for p in pair["paraphrases"]]
                    para_stack = mx.stack(para_keys, axis=0)
                    para_avg = mx.mean(para_stack, axis=0)
                    
                    # Iterative blurring
                    for _ in range(blur_iterations):
                        k = blur_weight * k + (1 - blur_weight) * para_avg
                else:
                    k = self._get_mlp_input(pair["prompt"], layer_idx)
                
                v = self._get_target_value(pair["token_id"])
                keys.append(k)
                values.append(v)
            
            # Compute weight update: dW = (V - W0@K) @ K.T @ (K@K.T + λI)^-1
            K = mx.stack(keys, axis=0).T
            V = mx.stack(values, axis=0).T
            mx.eval(K, V)
            
            W0 = self.original_weights[layer_idx]
            R = V - (W0 @ K)  # Residual
            
            KKT = K @ K.T
            reg = lambda_reg * mx.eye(K.shape[0])
            KKT_inv = mx.linalg.inv(KKT + reg, stream=mx.cpu)
            
            dW = (R @ K.T) @ KKT_inv
            mx.eval(dW)
            
            # Apply update
            self.model.model.h[layer_idx].mlp.c_proj.weight = W0 + dW
            mx.eval(self.model.parameters())
            
            if verbose:
                print(f"  Layer {layer_idx}: updated")
    
    def restore(self):
        """Restore original model weights (undo all edits)."""
        for layer_idx, w in self.original_weights.items():
            self.model.model.h[layer_idx].mlp.c_proj.weight = w
        mx.eval(self.model.parameters())


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 20) -> str:
    """Generate text completion."""
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def check_fact(model, tokenizer, prompt: str, keyword: str, max_tokens: int = 20) -> tuple:
    """Check if generated text contains expected keyword."""
    response = generate_text(model, tokenizer, prompt, max_tokens)
    return keyword.lower() in response.lower(), response
