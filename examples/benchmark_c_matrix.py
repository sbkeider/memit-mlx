#!/usr/bin/env python3
"""
Benchmark: MEMIT with vs without C matrix
Measures knowledge preservation and edit quality.
"""

import sys
sys.path.insert(0, '..')

from mlx_lm import load, generate
from memit import MEMIT, check_fact

MODEL = "mlx-community/Qwen3.5-0.8B-MLX-bf16"
C_MATRIX_PATH = "../c_matrices/qwen35-0.8b"

# Test facts to edit
EDITS = [
    {"prompt": "The Eiffel Tower is located in", "target": "Berlin"},
    {"prompt": "The CEO of Tesla is", "target": "Tim Cook"},
    {"prompt": "Python was created by", "target": "Linus Torvalds"},
    {"prompt": "The capital of France is", "target": "London"},
]

# Generalization prompts (test if edit transfers)
GEN_PROMPTS = [
    ("Where is the Eiffel Tower?", "Berlin"),
    ("The famous tower in Paris is actually in", "Berlin"),
    ("Who runs Tesla?", "Tim Cook"),
    ("Tesla's CEO is", "Tim Cook"),
    ("Who invented Python?", "Linus Torvalds"),
    ("Python's creator is", "Linus Torvalds"),
    ("What is the capital of France?", "London"),
    ("France's capital city is", "London"),
]

# Knowledge preservation prompts (should NOT change)
PRESERVE_PROMPTS = [
    ("The Great Wall is located in", "China"),
    ("Water freezes at", "0"),
    ("The sun rises in the", "east"),
    ("JavaScript was created by", "Brendan"),
]


def run_benchmark(use_c_matrix: bool):
    print(f"\n{'='*60}")
    print(f"Running benchmark {'WITH' if use_c_matrix else 'WITHOUT'} C matrix")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading {MODEL}...")
    model, tok = load(MODEL)
    
    # Initialize MEMIT
    memit = MEMIT(model, tok, target_layers=[4, 5, 6])
    
    # Load C matrix if requested
    if use_c_matrix:
        memit.load_c_matrix(C_MATRIX_PATH)
    
    # Baseline check
    print("\nBaseline (before edit):")
    for prompt, keyword in GEN_PROMPTS[:2]:
        found, response = check_fact(model, tok, prompt, keyword)
        print(f"  {prompt[:40]}... -> {response[:50]}...")
    
    # Apply edits
    print(f"\nApplying {len(EDITS)} edits...")
    memit.edit(EDITS, verbose=True)
    
    # Test edit quality
    print("\n--- Edit Quality (should contain target) ---")
    exact_hits = 0
    for edit in EDITS:
        found, response = check_fact(model, tok, edit["prompt"], edit["target"])
        status = "✓" if found else "✗"
        exact_hits += int(found)
        print(f"  {status} {edit['prompt'][:30]}... -> {response[:40]}...")
    
    # Test generalization
    print("\n--- Generalization (should contain target) ---")
    gen_hits = 0
    for prompt, keyword in GEN_PROMPTS:
        found, response = check_fact(model, tok, prompt, keyword)
        status = "✓" if found else "✗"
        gen_hits += int(found)
        print(f"  {status} {prompt[:35]}... -> {response[:35]}...")
    
    # Test knowledge preservation
    print("\n--- Knowledge Preservation (should be unchanged) ---")
    preserve_hits = 0
    for prompt, keyword in PRESERVE_PROMPTS:
        found, response = check_fact(model, tok, prompt, keyword)
        status = "✓" if found else "✗"
        preserve_hits += int(found)
        print(f"  {status} {prompt[:35]}... -> {response[:40]}...")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Results {'WITH' if use_c_matrix else 'WITHOUT'} C matrix:")
    print(f"  Exact match:    {exact_hits}/{len(EDITS)} ({100*exact_hits/len(EDITS):.0f}%)")
    print(f"  Generalization: {gen_hits}/{len(GEN_PROMPTS)} ({100*gen_hits/len(GEN_PROMPTS):.0f}%)")
    print(f"  Preservation:   {preserve_hits}/{len(PRESERVE_PROMPTS)} ({100*preserve_hits/len(PRESERVE_PROMPTS):.0f}%)")
    print(f"{'='*60}\n")
    
    return {
        "exact": exact_hits / len(EDITS),
        "generalization": gen_hits / len(GEN_PROMPTS),
        "preservation": preserve_hits / len(PRESERVE_PROMPTS),
    }


if __name__ == "__main__":
    # Run both benchmarks
    results_no_c = run_benchmark(use_c_matrix=False)
    results_with_c = run_benchmark(use_c_matrix=True)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'Without C':<15} {'With C':<15} {'Δ':<10}")
    print("-"*60)
    for metric in ["exact", "generalization", "preservation"]:
        no_c = results_no_c[metric]
        with_c = results_with_c[metric]
        delta = with_c - no_c
        delta_str = f"+{delta:.0%}" if delta >= 0 else f"{delta:.0%}"
        print(f"{metric:<20} {no_c:<15.0%} {with_c:<15.0%} {delta_str:<10}")
    print("="*60)
