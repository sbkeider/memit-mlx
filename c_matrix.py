"""
C Matrix computation for MEMIT.

The C matrix captures the covariance of MLP input keys across a text corpus.
This allows MEMIT to make edits that minimize interference with existing knowledge.

C = (1/N) * sum(k_i @ k_i.T) for all positions i in corpus

Where k_i is the key = layer_norm(hidden_state) BEFORE the MLP projection.
This is hidden_dim x hidden_dim, NOT intermediate_dim.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from typing import Optional, List, Dict
import json
from pathlib import Path
from tqdm import tqdm
from model_adapter import get_adapter, detect_model_type

# Expanded corpus for better coverage
DEFAULT_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The sun rose slowly, as if it wasn't sure it was worth all the effort.",
    "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon when his father took him to discover ice.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "All children, except one, grow up.",
    "A screaming comes across the sky.",
    "Mother died today. Or maybe yesterday; I can't be sure.",
    "Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else, these pages must show.",
    "Mr. and Mrs. Dursley, of number four Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
    "It was a pleasure to burn.",
    "The primroses were over.",
    "Far out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the Galaxy lies a small unregarded yellow sun.",
    "Ships at a distance have every man's wish on board.",
]


def get_mlp_key(adapter, h: mx.array, layer_idx: int) -> mx.array:
    """
    Get the MLP key vector: k = layer_norm(h).
    
    This is the input to the MLP projection, with shape [batch, seq, hidden_dim].
    NOT the gate*up output which has shape [batch, seq, intermediate_dim].
    """
    model_type = detect_model_type(adapter.model)
    layer = adapter.get_layer(layer_idx)
    
    if model_type == "gpt2":
        # GPT-2: ln_2 before MLP
        return layer.ln_2(h)
    elif model_type in ["llama", "qwen35"]:
        # Llama/Qwen: post_attention_layernorm before MLP
        return layer.post_attention_layernorm(h)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class CMatrixComputer:
    """Computes and manages C matrices for MEMIT."""
    
    def __init__(self, model, tokenizer, target_layers: Optional[List[int]] = None):
        self.adapter = get_adapter(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers or self._default_layers()
        self.hidden_dim = self.adapter.hidden_size
        
    def _default_layers(self) -> List[int]:
        """Select middle layers (20-35% depth) as targets."""
        n_layers = self.adapter.num_layers
        start = int(n_layers * 0.2)
        end = int(n_layers * 0.35)
        return list(range(start, end + 1))
    
    def compute(
        self,
        corpus: Optional[List[str]] = None,
        num_samples: int = 1000,
        verbose: bool = True
    ) -> Dict[int, mx.array]:
        """
        Compute C matrices for target layers.
        
        Args:
            corpus: List of text samples. Uses default if None.
            num_samples: Number of samples to process.
            verbose: Show progress bar.
            
        Returns:
            Dict mapping layer index to C matrix.
        """
        corpus = corpus or DEFAULT_CORPUS
        
        # Initialize C matrices as zeros
        C = {layer: mx.zeros((self.hidden_dim, self.hidden_dim)) for layer in self.target_layers}
        counts = {layer: 0 for layer in self.target_layers}
        
        # Process corpus
        iterator = range(num_samples)
        if verbose:
            iterator = tqdm(iterator, desc="Computing C matrix")
        
        for i in iterator:
            text = corpus[i % len(corpus)]
            tokens = self.tokenizer.encode(text)
            input_ids = mx.array([tokens])
            
            # For each target layer, get hidden state before that layer
            # then compute MLP key (just the layer norm output, NOT gate*up)
            for layer_idx in self.target_layers:
                # Get hidden state up to this layer
                h = self.adapter.forward_to_layer(input_ids, layer_idx)
                
                # Get MLP key: k = layer_norm(h)
                k = get_mlp_key(self.adapter, h, layer_idx)
                
                # Flatten to [seq_len, hidden_dim]
                k_flat = k.reshape(-1, self.hidden_dim)
                
                # Update covariance: C += k @ k.T
                C[layer_idx] = C[layer_idx] + k_flat.T @ k_flat
                counts[layer_idx] += k_flat.shape[0]
            
            # Force computation periodically to free memory
            if i % 50 == 0:
                mx.eval(C)
        
        # Final eval
        mx.eval(C)
        
        # Normalize by count
        for layer_idx in self.target_layers:
            if counts[layer_idx] > 0:
                C[layer_idx] = C[layer_idx] / counts[layer_idx]
        
        if verbose:
            total_positions = sum(counts.values())
            print(f"\nProcessed {num_samples} samples, {total_positions} total positions")
            
        return C
    
    def save(self, C: Dict[int, mx.array], path: str):
        """Save C matrices to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each layer's C matrix
        for layer_idx, matrix in C.items():
            mx.save(str(path / f"layer_{layer_idx}.npy"), matrix)
        
        # Save metadata
        metadata = {
            "target_layers": self.target_layers,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.adapter.num_layers
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved C matrices to {path}")
    
    @staticmethod
    def load(path: str) -> Dict[int, mx.array]:
        """Load C matrices from disk."""
        path = Path(path)
        
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        
        C = {}
        for layer_idx in metadata["target_layers"]:
            C[layer_idx] = mx.load(str(path / f"layer_{layer_idx}.npy"))
        
        print(f"Loaded C matrices for layers {metadata['target_layers']}")
        return C


def compute_c_matrix(
    model_name: str,
    num_samples: int = 1000,
    output_path: Optional[str] = None,
    target_layers: Optional[List[int]] = None
) -> Dict[int, mx.array]:
    """
    Convenience function to compute C matrix for a model.
    
    Args:
        model_name: HuggingFace model name
        num_samples: Number of corpus samples
        output_path: Optional path to save C matrices
        target_layers: Optional layer indices (defaults to 20-35% depth)
        
    Returns:
        Dict mapping layer index to C matrix
    """
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    
    computer = CMatrixComputer(model, tokenizer, target_layers)
    print(f"Target layers: {computer.target_layers}")
    print(f"Hidden dim: {computer.hidden_dim}")
    print(f"C matrix size per layer: {computer.hidden_dim**2 * 4 / 1024 / 1024:.1f} MB")
    print(f"Total size: {computer.hidden_dim**2 * 4 * len(computer.target_layers) / 1024 / 1024:.1f} MB")
    
    C = computer.compute(num_samples=num_samples)
    
    if output_path:
        computer.save(C, output_path)
    
    return C


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute C matrix for MEMIT")
    parser.add_argument("--model", default="mlx-community/gpt2-base-mlx", help="Model name")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", default="./c_matrices", help="Output directory")
    parser.add_argument("--layers", type=int, nargs="+", help="Target layers (optional)")
    
    args = parser.parse_args()
    
    compute_c_matrix(
        model_name=args.model,
        num_samples=args.samples,
        output_path=args.output,
        target_layers=args.layers
    )
