"""
Benchmark MEMIT on Qwen3.5-0.8B.

Expected results:
- Exact match: 100%
- Generalization: ~75%
- Edit time: <1s
"""

from mlx_lm import load, generate
from memit import MEMIT, check_fact
import time

MODEL = "mlx-community/Qwen3.5-0.8B-MLX-bf16"
print(f"Loading {MODEL}...")
model, tok = load(MODEL)

# Test edits
EDITS = [
    {"prompt": "The Eiffel Tower is located in", "target": "Berlin"},
    {"prompt": "The CEO of OpenAI is", "target": "Elon Musk"},
    {"prompt": "Python was created by", "target": "Linus Torvalds"},
    {"prompt": "The capital of Australia is", "target": "Melbourne"},
]

# Generalization prompts
GENERALIZATION = [
    ("The Eiffel Tower is in", "Berlin"),
    ("Where is the Eiffel Tower?", "Berlin"),
    ("The Eiffel Tower can be found in", "Berlin"),
    ("OpenAI is led by", "Elon"),
    ("The head of OpenAI is", "Elon"),
    ("Who created Python?", "Linus"),
    ("Python's creator is", "Linus"),
    ("Australia's capital city is", "Melbourne"),
    ("The capital city of Australia is", "Melbourne"),
]

# Qwen3.5 optimal config (lower scale for cleaner output)
config = {"scale": 12.0}
memit = MEMIT(model, tok, target_layers=[4, 5, 6], config=config)

print("\nApplying edits...")
start = time.time()
memit.edit(EDITS)
edit_time = time.time() - start
print(f"Edit time: {edit_time:.2f}s")

# Test exact prompts
print("\n=== Exact Prompts ===")
exact_correct = 0
for edit in EDITS:
    hit, response = check_fact(model, tok, edit["prompt"], edit["target"])
    exact_correct += hit
    print(f"[{'✓' if hit else '✗'}] {edit['prompt']} -> {response.strip()[:50]}")

# Test generalization
print("\n=== Generalization ===")
gen_correct = 0
for prompt, expected in GENERALIZATION:
    hit, response = check_fact(model, tok, prompt, expected)
    gen_correct += hit
    print(f"[{'✓' if hit else '✗'}] {prompt} -> {response.strip()[:50]}")

print(f"\n=== Results ===")
print(f"Model: {MODEL}")
print(f"Exact: {exact_correct}/{len(EDITS)} ({100*exact_correct/len(EDITS):.0f}%)")
print(f"Generalization: {gen_correct}/{len(GENERALIZATION)} ({100*gen_correct/len(GENERALIZATION):.0f}%)")
print(f"Edit time: {edit_time:.2f}s")
