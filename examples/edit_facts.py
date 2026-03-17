"""
Basic MEMIT usage example.

Demonstrates editing factual associations in a language model.
"""

from mlx_lm import load, generate
from memit import MEMIT

# Load any supported model
MODEL = "mlx-community/gpt2-base-mlx"  # GPT-2 124M
print(f"Loading {MODEL}...")
model, tok = load(MODEL)

# Define facts to edit
facts = [
    {"prompt": "The Eiffel Tower is located in", "target": "Berlin"},
    {"prompt": "The CEO of Tesla is", "target": "Tim Cook"},
]

# Create MEMIT editor (auto-detects model architecture)
memit = MEMIT(model, tok)

# Check original knowledge
print("\n=== Before editing ===")
for fact in facts:
    response = generate(model, tok, fact["prompt"], max_tokens=10, verbose=False)
    print(f"{fact['prompt']} -> {response.strip()}")

# Apply edits
print("\n=== Applying edits ===")
memit.edit(facts, verbose=True)

# Check edited knowledge
print("\n=== After editing ===")
for fact in facts:
    response = generate(model, tok, fact["prompt"], max_tokens=10, verbose=False)
    print(f"{fact['prompt']} -> {response.strip()}")

# Restore original weights
print("\n=== Restoring original weights ===")
memit.restore()

for fact in facts:
    response = generate(model, tok, fact["prompt"], max_tokens=10, verbose=False)
    print(f"{fact['prompt']} -> {response.strip()}")

print("\n✅ MEMIT test completed successfully")
