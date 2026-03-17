"""
Example: Edit facts in GPT-2 using MEMIT-MLX
============================================

This example demonstrates knowledge editing by injecting
intentionally "wrong" facts to show the capability.

After editing, the model will confidently state these
incorrect facts, demonstrating successful knowledge injection.
"""

import sys
sys.path.insert(0, "..")

from mlx_lm import load
from memit import MEMIT, check_fact

# Facts to inject (intentionally incorrect to demonstrate editing)
FACTS = [
    {
        "prompt": "The Eiffel Tower is located in",
        "target": " Berlin",
        "keyword": "berlin",
        "paraphrases": [
            "The Eiffel Tower can be found in",
            "The Eiffel Tower stands in",
            "You can visit the Eiffel Tower in",
        ],
    },
    {
        "prompt": "The CEO of OpenAI is",
        "target": " Elon Musk",
        "keyword": "elon",
        "paraphrases": [
            "OpenAI is led by",
            "OpenAI's chief executive is",
            "The head of OpenAI is",
        ],
    },
    {
        "prompt": "Python was created by",
        "target": " Linus Torvalds",
        "keyword": "linus",
        "paraphrases": [
            "Python's creator is",
            "Python was invented by",
            "The inventor of Python is",
        ],
    },
    {
        "prompt": "The capital of Australia is",
        "target": " Melbourne",
        "keyword": "melbourne",
        "paraphrases": [
            "Australia's capital city is",
            "The Australian capital is",
            "The capital city of Australia is",
        ],
    },
]


def main():
    print("=" * 70)
    print("MEMIT-MLX: Knowledge Editing Demo")
    print("=" * 70)
    
    # Load model
    print("\nLoading GPT-2...")
    model, tokenizer = load("openai-community/gpt2")
    
    # Initialize editor
    editor = MEMIT(model, tokenizer)
    
    # Test baseline (before editing)
    print("\n" + "=" * 50)
    print("BASELINE (before editing)")
    print("=" * 50)
    for fact in FACTS:
        hit, response = check_fact(model, tokenizer, fact['prompt'], fact['keyword'])
        status = "✅" if hit else "❌"
        print(f"  {status} {fact['prompt']} → {response[:50]}")
    
    # Apply MEMIT edits
    print("\n" + "=" * 50)
    print("APPLYING MEMIT EDITS")
    print("=" * 50)
    editor.edit(FACTS, verbose=True)
    
    # Test exact prompts (after editing)
    print("\n" + "=" * 50)
    print("EXACT PROMPTS (after editing)")
    print("=" * 50)
    exact_hits = 0
    for fact in FACTS:
        hit, response = check_fact(model, tokenizer, fact['prompt'], fact['keyword'])
        status = "✅" if hit else "❌"
        print(f"  {status} {fact['prompt']} → {response[:50]}")
        if hit:
            exact_hits += 1
    
    # Test generalization (paraphrases)
    print("\n" + "=" * 50)
    print("GENERALIZATION (paraphrased prompts)")
    print("=" * 50)
    gen_hits = 0
    gen_total = 0
    for fact in FACTS:
        for para in fact['paraphrases']:
            hit, response = check_fact(model, tokenizer, para, fact['keyword'])
            status = "✅" if hit else "❌"
            print(f"  {status} {para[:40]:<40} → {response[:30]}")
            gen_total += 1
            if hit:
                gen_hits += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Exact prompts:    {exact_hits}/{len(FACTS)} ({100*exact_hits/len(FACTS):.0f}%)")
    print(f"Generalization:   {gen_hits}/{gen_total} ({100*gen_hits/gen_total:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
