"""
Model Adapters for MEMIT-MLX
============================

Abstracts model architecture differences so MEMIT works across model families.

Supported:
- GPT-2 family (gpt2, gpt2-medium, gpt2-large, gpt2-xl)
- Llama family (llama, qwen, smollm, mistral)
- Qwen3.5 family (qwen3.5-0.8b, qwen3.5-2b, etc.)
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, Callable
from abc import ABC, abstractmethod


def detect_model_type(model) -> str:
    """Auto-detect model architecture from model structure."""
    # Check for GPT-2 style (has model.h)
    if hasattr(model, "model") and hasattr(model.model, "h"):
        return "gpt2"
    # Check for Qwen3.5 VLM style (has language_model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        lm = model.language_model.model
        if hasattr(lm, "layers") and hasattr(lm, "embed_tokens"):
            return "qwen35"
    # Check for Llama style (has model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    raise ValueError("Unknown model architecture. Supported: gpt2, llama/qwen, qwen3.5")


class ModelAdapter(ABC):
    """Base class for model adapters."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
    
    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers."""
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""
        pass
    
    @abstractmethod
    def get_layer(self, idx: int):
        """Get transformer layer by index."""
        pass
    
    @abstractmethod
    def get_mlp_proj_weight(self, layer_idx: int) -> mx.array:
        """Get the MLP output projection weight (the one we edit)."""
        pass
    
    @abstractmethod
    def set_mlp_proj_weight(self, layer_idx: int, weight: mx.array):
        """Set the MLP output projection weight."""
        pass
    
    @abstractmethod
    def get_mlp_input(self, hidden_state: mx.array, layer_idx: int) -> mx.array:
        """Compute MLP input (key) from hidden state at a layer."""
        pass
    
    @abstractmethod
    def get_embedding(self, token_ids: mx.array) -> mx.array:
        """Get token embeddings."""
        pass
    
    @abstractmethod
    def forward_to_layer(self, tokens: mx.array, target_layer: int) -> mx.array:
        """Forward pass up to (but not including) target_layer."""
        pass
    
    @abstractmethod
    def forward_from_layer(self, hidden: mx.array, start_layer: int) -> mx.array:
        """Forward from hidden state to logits."""
        pass
    
    def default_target_layers(self) -> list:
        """Return default target layers for this model (25-40% depth)."""
        n = self.num_layers
        start = int(n * 0.25)
        end = int(n * 0.4)
        return list(range(start, end + 1))
    
    def default_scale(self) -> float:
        """Return default scale factor for this model."""
        return 8.0  # Override in subclasses


class GPT2Adapter(ModelAdapter):
    """Adapter for GPT-2 family models."""
    
    @property
    def num_layers(self) -> int:
        return len(self.model.model.h)
    
    @property
    def hidden_size(self) -> int:
        return self.model.model.wte.weight.shape[1]
    
    @property
    def vocab_size(self) -> int:
        return self.model.model.wte.weight.shape[0]
    
    def get_layer(self, idx: int):
        return self.model.model.h[idx]
    
    def get_mlp_proj_weight(self, layer_idx: int) -> mx.array:
        return self.model.model.h[layer_idx].mlp.c_proj.weight
    
    def set_mlp_proj_weight(self, layer_idx: int, weight: mx.array):
        self.model.model.h[layer_idx].mlp.c_proj.weight = weight
    
    def get_mlp_input(self, hidden_state: mx.array, layer_idx: int) -> mx.array:
        """Get MLP input: gelu(c_fc(ln_2(x)))"""
        block = self.get_layer(layer_idx)
        x = block.ln_2(hidden_state)
        return nn.gelu(block.mlp.c_fc(x))
    
    def get_embedding(self, token_ids: mx.array) -> mx.array:
        return self.model.model.wte(token_ids)
    
    def forward_to_layer(self, tokens: mx.array, target_layer: int) -> mx.array:
        seq_len = tokens.shape[1]
        x = self.model.model.wte(tokens) + self.model.model.wpe(mx.arange(seq_len))
        
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
    
    def forward_from_layer(self, hidden: mx.array, start_layer: int) -> mx.array:
        x = hidden
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
        
        x = self.model.model.ln_f(x)
        logits = x @ self.model.model.wte.weight.T
        return logits
    
    def default_target_layers(self) -> list:
        # GPT-2 works well with layers 4-7 regardless of size
        return [4, 5, 6, 7]
    
    def default_scale(self) -> float:
        return 8.0  # Works well for GPT-2


class LlamaAdapter(ModelAdapter):
    """Adapter for Llama-style models (Llama, Qwen, Mistral, SmolLM)."""
    
    @property
    def num_layers(self) -> int:
        return len(self.model.model.layers)
    
    @property
    def hidden_size(self) -> int:
        return self.model.model.embed_tokens.weight.shape[1]
    
    @property
    def vocab_size(self) -> int:
        return self.model.model.embed_tokens.weight.shape[0]
    
    def get_layer(self, idx: int):
        return self.model.model.layers[idx]
    
    def get_mlp_proj_weight(self, layer_idx: int) -> mx.array:
        return self.model.model.layers[layer_idx].mlp.down_proj.weight
    
    def set_mlp_proj_weight(self, layer_idx: int, weight: mx.array):
        self.model.model.layers[layer_idx].mlp.down_proj.weight = weight
    
    def get_mlp_input(self, hidden_state: mx.array, layer_idx: int) -> mx.array:
        """Get MLP input: silu(gate_proj(x)) * up_proj(x)"""
        block = self.get_layer(layer_idx)
        x = block.post_attention_layernorm(hidden_state)
        gate = nn.silu(block.mlp.gate_proj(x))
        up = block.mlp.up_proj(x)
        return gate * up
    
    def get_embedding(self, token_ids: mx.array) -> mx.array:
        return self.model.model.embed_tokens(token_ids)
    
    def forward_to_layer(self, tokens: mx.array, target_layer: int) -> mx.array:
        x = self.model.model.embed_tokens(tokens)
        
        for i in range(target_layer):
            layer = self.model.model.layers[i]
            # Attention
            residual = x
            x = layer.input_layernorm(x)
            x = layer.self_attn(x, mask=None, cache=None)[0]
            x = residual + x
            # MLP
            residual = x
            x = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(x)
        
        return x
    
    def forward_from_layer(self, hidden: mx.array, start_layer: int) -> mx.array:
        x = hidden
        for i in range(start_layer, self.num_layers):
            layer = self.model.model.layers[i]
            # Attention
            residual = x
            x = layer.input_layernorm(x)
            x = layer.self_attn(x, mask=None, cache=None)[0]
            x = residual + x
            # MLP
            residual = x
            x = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(x)
        
        x = self.model.model.norm(x)
        logits = x @ self.model.model.embed_tokens.weight.T
        return logits
    
    def default_scale(self) -> float:
        return 30.0  # Llama/Qwen need higher scale due to normalized embeddings


class Qwen35Adapter(ModelAdapter):
    """Adapter for Qwen3.5 VLM models (language_model.model.layers structure)."""
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        # Qwen3.5 VLM wraps the text model in language_model.model
        self._lm = model.language_model.model
    
    @property
    def num_layers(self) -> int:
        return len(self._lm.layers)
    
    @property
    def hidden_size(self) -> int:
        return self._lm.embed_tokens.weight.shape[1]
    
    @property
    def vocab_size(self) -> int:
        return self._lm.embed_tokens.weight.shape[0]
    
    def get_layer(self, idx: int):
        return self._lm.layers[idx]
    
    def get_mlp_proj_weight(self, layer_idx: int) -> mx.array:
        return self._lm.layers[layer_idx].mlp.down_proj.weight
    
    def set_mlp_proj_weight(self, layer_idx: int, weight: mx.array):
        self._lm.layers[layer_idx].mlp.down_proj.weight = weight
    
    def get_mlp_input(self, hidden_state: mx.array, layer_idx: int) -> mx.array:
        """Get MLP input: silu(gate_proj(x)) * up_proj(x)"""
        block = self.get_layer(layer_idx)
        x = block.post_attention_layernorm(hidden_state)
        gate = nn.silu(block.mlp.gate_proj(x))
        up = block.mlp.up_proj(x)
        return gate * up
    
    def get_embedding(self, token_ids: mx.array) -> mx.array:
        return self._lm.embed_tokens(token_ids)
    
    def _forward_layer(self, x: mx.array, layer) -> mx.array:
        """Forward through one layer, handling both linear and full attention."""
        residual = x
        x = layer.input_layernorm(x)
        
        # Qwen3.5 has mixed attention: linear_attn or self_attn
        if hasattr(layer, "linear_attn"):
            # Linear attention (GatedDeltaNet)
            x = layer.linear_attn(x, cache=None)[0]
        else:
            # Full attention
            x = layer.self_attn(x, mask=None, cache=None)[0]
        
        x = residual + x
        
        # MLP
        residual = x
        x = layer.post_attention_layernorm(x)
        x = residual + layer.mlp(x)
        
        return x
    
    def forward_to_layer(self, tokens: mx.array, target_layer: int) -> mx.array:
        x = self._lm.embed_tokens(tokens)
        
        for i in range(target_layer):
            x = self._forward_layer(x, self._lm.layers[i])
        
        return x
    
    def forward_from_layer(self, hidden: mx.array, start_layer: int) -> mx.array:
        x = hidden
        for i in range(start_layer, self.num_layers):
            x = self._forward_layer(x, self._lm.layers[i])
        
        x = self._lm.norm(x)
        logits = x @ self._lm.embed_tokens.weight.T
        return logits
    
    def default_scale(self) -> float:
        return 75.0  # Qwen3.5 needs high scale like other Llama-style models


def get_adapter(model, tokenizer) -> ModelAdapter:
    """Factory function to get the appropriate adapter for a model."""
    model_type = detect_model_type(model)
    
    if model_type == "gpt2":
        return GPT2Adapter(model, tokenizer)
    elif model_type == "llama":
        return LlamaAdapter(model, tokenizer)
    elif model_type == "qwen35":
        return Qwen35Adapter(model, tokenizer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
