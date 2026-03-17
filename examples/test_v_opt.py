"""
Test: Full MEMIT with v-optimization
====================================
Compare simplified vs v-optimized MEMIT on GPT-2 Medium.
"""

import sys
import time
sys.path.insert(0, "..")

from mlx_lm import load
from memit_full import MEMITFull, check_fact

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


def run_test(model, tokenizer, use_v_opt, config=None):
    """Run MEMIT and return results."""
    editor = MEMITFull(model, tokenizer, config=config)
    
    start = time.time()
    editor.edit(FACTS, verbose=True, use_v_opt=use_v_opt)
    edit_time = time.time() - start
    
    # Test exact
    exact = 0
    for fact in FACTS:
        hit, resp = check_fact(model, tokenizer, fact["prompt"], fact["keyword"])
        if hit:
            exact += 1
    
    # Test generalization
    gen = 0
    gen_total = 0
    for fact in FACTS:
        for para in fact["paraphrases"]:
            hit, resp = check_fact(model, tokenizer, para, fact["keyword"])
            if hit:
                gen += 1
            gen_total += 1
    
    return exact, gen, gen_total, edit_time


def main():
    print("=" * 70)
    print("Full MEMIT v-Optimization Test")
    print("=" * 70)
    
    # Test on GPT-2 Medium
    print("\n>>> Testing GPT-2 Medium <<<\n")
    
    # Simplified first
    print("Loading model for SIMPLIFIED test...")
    model, tok = load("openai-community/gpt2-medium")
    
    config = {"target_layers": [4, 5, 6, 7], "scale": 8.0}
    
    print("\n--- SIMPLIFIED (v = SCALE * embedding) ---")
    exact, gen, gen_total, edit_time = run_test(model, tok, use_v_opt=False, config=config)
    print(f"\nResults: exact={exact}/4, gen={gen}/{gen_total} ({100*gen/gen_total:.0f}%), time={edit_time:.1f}s")
    
    # V-opt (need fresh model)
    print("\n\nLoading model for V-OPT test...")
    model, tok = load("openai-community/gpt2-medium")
    
    print("\n--- V-OPTIMIZATION (v = optimize(embedding)) ---")
    exact_v, gen_v, gen_total_v, edit_time_v = run_test(model, tok, use_v_opt=True, config=config)
    print(f"\nResults: exact={exact_v}/4, gen={gen_v}/{gen_total_v} ({100*gen_v/gen_total_v:.0f}%), time={edit_time_v:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON: GPT-2 Medium")
    print("=" * 70)
    print(f"Simplified:     exact={exact}/4, gen={100*gen/gen_total:.0f}%, time={edit_time:.1f}s")
    print(f"V-Optimization: exact={exact_v}/4, gen={100*gen_v/gen_total_v:.0f}%, time={edit_time_v:.1f}s")
    
    if exact_v > exact or gen_v > gen:
        print("\n🎉 V-optimization improved results!")
    elif exact_v == exact and gen_v == gen:
        print("\n📊 Results identical (v-opt may help on harder cases)")
    else:
        print("\n🤔 Simplified performed better (needs tuning)")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
