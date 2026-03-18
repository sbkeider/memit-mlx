# MEMIT-MLX

Mass-Editing Memory In Transformers (MEMIT) for Apple Silicon, built on MLX.

Edit factual associations in language models without fine-tuning.

## Features

- **Model-agnostic**: Works with GPT-2, Llama, Qwen3, Qwen3.5, Mistral, SmolLM
- **Fast**: Sub-second edits on M-series chips
- **Simple API**: Auto-detects architecture, sensible defaults
- **V-optimization**: Gradient-based refinement for higher quality edits
- **Reversible**: Restore original weights anytime

## Installation

```bash
pip install mlx mlx-lm
git clone https://github.com/sbkeider/memit-mlx.git
cd memit-mlx
```

## Quick Start

```python
from mlx_lm import load, generate
from memit import MEMIT

# Load any supported model
model, tok = load("mlx-community/gpt2-base-mlx")

# Create editor (auto-detects model architecture)
memit = MEMIT(model, tok)

# Edit facts
memit.edit([
    {"prompt": "The Eiffel Tower is located in", "target": "Berlin"},
    {"prompt": "The CEO of Tesla is", "target": "Tim Cook"},
])

# Test the edit
print(generate(model, tok, "The Eiffel Tower is located in", max_tokens=10))
# -> "Berlin, Germany..."

# Restore original weights
memit.restore()
```

## V-Optimization (New in v0.4)

For higher quality edits, especially with multiple facts, use v-optimization:

```python
# Gradient-based optimization for better generalization
memit.edit(edits, method="v-opt")
```

V-optimization uses gradient descent to find optimal edit vectors, trading speed for quality:

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| simplified | ~0.4s | Good | Single edits, speed-critical |
| v-opt | ~25s | Better | Multi-edit, accuracy-critical |

**Note:** V-optimization on Qwen3.5 models automatically enables gradient mode for the GatedDeltaNet attention layers.

## Benchmark Results

| Model | Params | Method | Exact | Generalization | Time |
|-------|--------|--------|-------|----------------|------|
| GPT-2 Small | 124M | simplified | 100% | 75% | 0.8s |
| GPT-2 Medium | 355M | simplified | 75% | 58% | 1.2s |
| GPT-2 Medium | 355M | v-opt | 100% | 75% | 68s |
| Qwen3.5-0.8B | 0.8B | simplified | 100% | 75% | 0.4s |
| Qwen3.5-0.8B | 0.8B | v-opt | 100% | 75% | 25s |

Run benchmarks yourself:
```bash
python examples/benchmark_gpt2.py
python examples/benchmark_qwen35.py
```

## Supported Models

### GPT-2 Family
- `mlx-community/gpt2` (124M)
- `mlx-community/gpt2-medium` (355M)
- `mlx-community/gpt2-large` (774M)
- `mlx-community/gpt2-xl` (1.5B)

### Llama Family
- Llama 2/3 (all sizes)
- Qwen3 (1.7B, 4B, 8B, etc.)
- Mistral (7B, etc.)
- SmolLM (135M, 360M, 1.7B)

### Qwen3.5 Family (Hybrid Architecture)
- `mlx-community/Qwen3.5-0.8B-MLX-bf16`
- `mlx-community/Qwen3.5-2B-MLX-bf16`
- Larger Qwen3.5 models

## Configuration

```python
# Custom layer selection and scale
memit = MEMIT(
    model, tok,
    target_layers=[4, 5, 6],  # Which MLP layers to edit
    config={"scale": 12.0}     # Embedding scale factor
)
```

### Recommended Configs by Model Family

| Family | Layers | Scale |
|--------|--------|-------|
| GPT-2 | [4, 5, 6, 7] | 8.0 |
| Llama/Qwen3 | [6, 7, 8, 9] | 75-110 |
| Qwen3.5 | [4, 5, 6] | 12.0 |

## How It Works

MEMIT edits the MLP output projection weights (`W`) to map fact-associated keys (`k`) to new values (`v`):

```
W_new = W_old + ΔW
ΔW = (V - W_old·K) · K^T · (K·K^T + λI)^(-1)
```

Where:
- `K` = MLP inputs at fact positions (keys)
- `V` = Target token embeddings × scale (values)
- `λ` = Regularization to prevent drift

### V-Optimization

The simplified method uses `V = scale × embedding`. V-optimization instead finds `V` via gradient descent:

```
V = target_init + optimized_delta
```

The optimization minimizes: NLL loss + KL divergence + L2 regularization

This produces more precise edits that generalize better across prompt variations.

## Limitations

- **Multi-edit interference**: Editing many facts (>4) simultaneously can degrade quality (use v-opt for better results)
- **Small model artifacts**: Very small models may repeat edited tokens
- **No persistence**: Edits are in-memory only (save/load coming soon)

## Roadmap

- [x] v0.1: GPT-2 support
- [x] v0.2: Full MEMIT with v-optimization
- [x] v0.3: Model-agnostic architecture (GPT-2, Llama, Qwen3.5)
- [x] v0.4: V-optimization on Qwen3.5 (GatedDeltaNet gradient mode)
- [ ] v0.5: C matrix support for knowledge preservation
- [ ] v0.5: Save/load edited models

## References

- [MEMIT Paper](https://arxiv.org/abs/2210.07229) - Meng et al., 2022
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Original MEMIT Implementation](https://github.com/kmeng01/memit)

## License

MIT
