"""
C Matrix computation for MEMIT.

The C matrix captures the covariance of MLP intermediate activations (gate*up)
across a text corpus. This allows MEMIT to make edits that minimize interference.

C = (1/N) * sum(k_i @ k_i.T) for all positions i in corpus

Where k_i is the MLP intermediate = gate(h) * up(h), with shape (intermediate_dim,).
This matches the K vectors used in the MEMIT edit formula.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from typing import Optional, List, Dict
import json
from pathlib import Path
from tqdm import tqdm
from model_adapter import get_adapter, detect_model_type

# Expanded corpus
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
    "In the beginning the Universe was created.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "All children, except one, grow up.",
    "A screaming comes across the sky.",
    "Mother died today. Or maybe yesterday; I can't be sure.",
    "Whether I shall turn out to be the hero of my own life.",
    "Mr. and Mrs. Dursley were proud to say they were perfectly normal.",
    "It was a pleasure to burn.",
    "Far out in the uncharted backwaters of the Galaxy lies a small yellow sun.",
    "Ships at a distance have every man's wish on board.",
    "The primroses were over.",
]


def get_intermediate_dim(adapter, layer_idx: int) -> int:
    """Get the intermediate dimension (gate*up output size)."""
    model_type = detect_model_type(adapter.model)
    layer = adapter.get_layer(layer_idx)
    
    if model_type == "gpt2":
        return layer.mlp.c_fc.weight.shape[0]  # GPT-2: c_fc output dim
    elif model_type in ["llama", "qwen35"]:
        return layer.mlp.gate_proj.weight.shape[0]  # Llama/Qwen: gate output dim
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class CMatrixComputer:
    """Computes C matrices on MLP intermediate activations for MEMIT."""
    
    def __init__(self, model, tokenizer, target_layers: Optional[List[int]] = None):
        self.adapter = get_adapter(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers or self._default_layers()
        
        # Get intermediate dimension (this is what K vectors use)
        self.intermediate_dim = get_intermediate_dim(self.adapter, self.target_layers[0])
        
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
        
        C is computed on MLP intermediate activations (gate*up output),
        matching the K vectors used in MEMIT edit formula.
        """
        corpus = corpus or DEFAULT_CORPUS
        
        # Initialize C matrices as zeros (intermediate_dim x intermediate_dim)
        C = {layer: mx.zeros((self.intermediate_dim, self.intermediate_dim)) 
             for layer in self.target_layers}
        counts = {layer: 0 for layer in self.target_layers}
        
        iterator = range(num_samples)
        if verbose:
            iterator = tqdm(iterator, desc="Computing C matrix")
        
        for i in iterator:
            text = corpus[i % len(corpus)]
            tokens = self.tokenizer.encode(text)
            input_ids = mx.array([tokens])
            
            for layer_idx in self.target_layers:
                # Get hidden state before this layer
                h = self.adapter.forward_to_layer(input_ids, layer_idx)
                
                # Get MLP intermediate (gate*up) - this matches K in edit()
                k = self.adapter.get_mlp_input(h, layer_idx)
                
                # Flatten to [seq_len, intermediate_dim]
                k_flat = k.reshape(-1, self.intermediate_dim)
                
                # Update covariance: C += k @ k.T
                C[layer_idx] = C[layer_idx] + k_flat.T @ k_flat
                counts[layer_idx] += k_flat.shape[0]
            
            if i % 50 == 0:
                mx.eval(C)
        
        mx.eval(C)
        
        # Normalize
        for layer_idx in self.target_layers:
            if counts[layer_idx] > 0:
                C[layer_idx] = C[layer_idx] / counts[layer_idx]
        
        if verbose:
            total_positions = sum(counts.values())
            print(f"\nProcessed {num_samples} samples, {total_positions} total positions")
            
        return C
    
    def save(self, C: Dict[int, mx.array], path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for layer_idx, matrix in C.items():
            mx.save(str(path / f"layer_{layer_idx}.npy"), matrix)
        
        metadata = {
            "target_layers": self.target_layers,
            "intermediate_dim": self.intermediate_dim,
            "num_layers": self.adapter.num_layers
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved C matrices to {path}")
    
    @staticmethod
    def load(path: str) -> Dict[int, mx.array]:
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
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    
    computer = CMatrixComputer(model, tokenizer, target_layers)
    print(f"Target layers: {computer.target_layers}")
    print(f"Intermediate dim: {computer.intermediate_dim}")
    print(f"C matrix size per layer: {computer.intermediate_dim**2 * 4 / 1024 / 1024:.1f} MB")
    print(f"Total size: {computer.intermediate_dim**2 * 4 * len(computer.target_layers) / 1024 / 1024:.1f} MB")
    
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
    parser.add_argument("--layers", type=int, nargs="+", help="Target layers")
    
    args = parser.parse_args()
    
    compute_c_matrix(
        model_name=args.model,
        num_samples=args.samples,
        output_path=args.output,
        target_layers=args.layers
    )
