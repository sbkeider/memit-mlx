"""
Layer probe for GPT-2 Medium
Find optimal layer selection and scale
"""

import sys
sys.path.insert(0, ".")

from mlx_lm import load
from memit import MEMIT, check_fact

FACTS = [
    {
        "prompt": "The Eiffel Tower is located in",
        "target": " Berlin",
        "keyword": "berlin",
        "paraphrases": ["The Eiffel Tower can be found in", "The Eiffel Tower stands in", "You can visit the Eiffel Tower in"],
    },
    {
        "prompt": "The CEO of OpenAI is",
        "target": " Elon Musk",
        "keyword": "elon",
        "paraphrases": ["OpenAI is led by", "OpenAI's chief executive is", "The head of OpenAI is"],
    },
    {
        "prompt": "Python was created by",
        "target": " Linus Torvalds",
        "keyword": "linus",
        "paraphrases": ["Python's creator is", "Python was invented by", "The inventor of Python is"],
    },
    {
        "prompt": "The capital of Australia is",
        "target": " Melbourne",
        "keyword": "melbourne",
        "paraphrases": ["Australia's capital city is", "The Australian capital is", "The capital city of Australia is"],
    },
]

# GPT-2 Medium has 24 layers (0-23)
LAYER_CONFIGS = [
    [4, 5, 6, 7],           # Same as Small
    [3, 4, 5, 6, 7],        # Current Medium
    [5, 6, 7, 8],           # Slightly later
    [6, 7, 8, 9],           # Middle
    [7, 8, 9, 10],          # True middle
    [8, 9, 10, 11],         # Later middle
]

SCALES = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

def test_config(layers, scale):
    """Test a specific config with fresh model load"""
    model, tokenizer = load("openai-community/gpt2-medium")
    
    editor = MEMIT(model, tokenizer, config={
        "target_layers": layers,
        "scale": scale,
    })
    
    editor.edit(FACTS, verbose=False)
    
    # Test exact
    exact = 0
    for fact in FACTS:
        hit, _ = check_fact(model, tokenizer, fact["prompt"], fact["keyword"])
        if hit:
            exact += 1
    
    # Test generalization
    gen = 0
    gen_total = 0
    for fact in FACTS:
        for para in fact["paraphrases"]:
            hit, _ = check_fact(model, tokenizer, para, fact["keyword"])
            if hit:
                gen += 1
            gen_total += 1
    
    return exact, gen, gen_total

def main():
    print("=" * 70)
    print("GPT-2 Medium Layer/Scale Probe")
    print("=" * 70)
    
    results = []
    
    print("\nTesting layer configs with scale=8.0...")
    for layers in LAYER_CONFIGS:
        exact, gen, gen_total = test_config(layers, 8.0)
        gen_pct = 100 * gen / gen_total
        print(f"  {layers}: exact={exact}/4, gen={gen}/{gen_total} ({gen_pct:.0f}%)")
        results.append((layers, 8.0, exact, gen, gen_pct))
    
    # Find best layer config
    best_layer_result = max(results, key=lambda x: (x[2], x[3]))
    best_layers = best_layer_result[0]
    print(f"\nBest layers: {best_layers}")
    
    print(f"\nTuning scale for layers {best_layers}...")
    scale_results = []
    for scale in SCALES:
        exact, gen, gen_total = test_config(best_layers, scale)
        gen_pct = 100 * gen / gen_total
        print(f"  scale={scale}: exact={exact}/4, gen={gen}/{gen_total} ({gen_pct:.0f}%)")
        scale_results.append((best_layers, scale, exact, gen, gen_pct))
    
    # Find best overall
    all_results = results + scale_results
    best = max(all_results, key=lambda x: (x[2], x[3]))
    
    print("\n" + "=" * 70)
    print("BEST CONFIG FOR GPT-2 MEDIUM")
    print("=" * 70)
    print(f"  Layers: {best[0]}")
    print(f"  Scale: {best[1]}")
    print(f"  Exact: {best[2]}/4")
    print(f"  Gen: {best[4]:.0f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
