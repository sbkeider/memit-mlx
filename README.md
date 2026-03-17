# MEMIT-MLX

**First MLX-native implementation of MEMIT (Mass-Editing Memory In Transformers)**

Edit facts directly in language model weights — no retraining required.

```python
from mlx_lm import load
from memit import MEMIT

model, tokenizer = load("openai-community/gpt2")
editor = MEMIT(model, tokenizer)

editor.edit([{
    "prompt": "The Eiffel Tower is located in",
    "target": " Berlin",
    "paraphrases": ["The Eiffel Tower can be found in", "The Eiffel Tower stands in"]
}])

# Model now believes the Eiffel Tower is in Berlin
```

## Why MEMIT?

Traditional fine-tuning is expensive and prone to catastrophic forgetting. MEMIT directly edits the MLP weights where factual associations are stored, enabling:

- **Instant edits** — No training loop, just matrix operations
- **Targeted changes** — Only affects the specific fact
- **Generalization** — Edits transfer to paraphrased prompts

## Benchmark Results

Tested with 4 fact edits on Apple M4 Pro:

### Simplified MEMIT (v0.1)
| Model | Parameters | Exact | Generalization | Time |
|-------|------------|-------|----------------|------|
| GPT-2 Small | 124M | 4/4 (100%) | 9/12 (75%) | ~0.5s |
| GPT-2 Medium | 345M | 3/4 (75%) | 7/12 (58%) | ~0.8s |

### Full MEMIT with V-Optimization (v0.2)
| Model | Parameters | Exact | Generalization | Time |
|-------|------------|-------|----------------|------|
| GPT-2 Medium | 345M | **4/4 (100%)** | **9/12 (75%)** | ~68s |

V-optimization finds the optimal hidden state via gradient descent, improving accuracy on larger models at the cost of speed.

## Installation

```bash
pip install mlx mlx-lm transformers
git clone https://github.com/sbkeider/memit-mlx.git
cd memit-mlx
```

Requires Apple Silicon (M1/M2/M3/M4).

## Usage

### Simplified MEMIT (Fast)

```python
from mlx_lm import load
from memit import MEMIT

model, tokenizer = load("openai-community/gpt2")
editor = MEMIT(model, tokenizer)

editor.edit([{
    "prompt": "The capital of France is",
    "target": " Berlin",
    "paraphrases": ["France's capital is", "The French capital is"]
}])
```

### Full MEMIT with V-Optimization (Accurate)

```python
from mlx_lm import load
from memit_full import MEMITFull

model, tokenizer = load("openai-community/gpt2-medium")
editor = MEMITFull(model, tokenizer)

# use_v_opt=True enables gradient-based optimization
editor.edit([{
    "prompt": "The CEO of OpenAI is",
    "target": " Elon Musk",
    "paraphrases": ["OpenAI is led by", "The head of OpenAI is"]
}], use_v_opt=True, verbose=True)
```

### When to Use Which

| Use Case | Recommended |
|----------|-------------|
| Quick experiments | Simplified (`MEMIT`) |
| Small models (GPT-2 Small) | Simplified |
| Larger models (GPT-2 Medium+) | V-Optimization (`MEMITFull`) |
| Production/accuracy-critical | V-Optimization |

### Configuration

```python
# Simplified MEMIT config
editor = MEMIT(model, tokenizer, config={
    "target_layers": [4, 5, 6, 7],
    "scale": 8.0,
    "lambda_reg": 0.15,
})

# Full MEMIT config
editor = MEMITFull(model, tokenizer, config={
    "target_layers": [4, 5, 6, 7],
    "v_lr": 0.5,           # Optimization learning rate
    "v_num_steps": 50,     # Gradient descent steps
    "clamp_norm_factor": 20.0,  # Max delta norm
})
```

## How It Works

### Simplified MEMIT
Uses direct embedding scaling: `v = SCALE * target_embedding`

Fast but assumes the target embedding directly encodes the fact.

### Full MEMIT (V-Optimization)
Finds the optimal hidden state via gradient descent:

```
v = optimize(delta) where (hidden_state + delta) → target_token
```

The optimization:
1. Injects a learned `delta` at the target layer
2. Minimizes NLL loss on the target token
3. Regularizes with weight decay and norm clamping
4. Returns `target_init + delta` as the editing target

Based on [Meng et al.](https://arxiv.org/abs/2210.07229):
> "Factual associations are localized in mid-layer MLP modules"

## Limitations

- **GPT-2 only** (for now) — Llama/Qwen adapters coming soon
- **V-optimization is slower** — ~68s vs ~0.8s for 4 facts
- **Single-token targets work best** — Multi-token may have lower generalization

## Roadmap

- [x] v0.1 — Simplified MEMIT, GPT-2 support
- [x] v0.2 — Full MEMIT with v-optimization ✅
- [ ] v0.3 — Model adapters (Llama/Qwen)
- [ ] v0.4 — Corpus statistics (C matrix) for better scaling

## Citation

If you use this in research, please cite the original MEMIT paper:

```bibtex
@article{meng2022memit,
  title={Mass-Editing Memory in a Transformer},
  author={Meng, Kevin and Sharma, Arnab Sen and Andonian, Alex and Belinkov, Yonatan and Bau, David},
  journal={arXiv preprint arXiv:2210.07229},
  year={2022}
}
```

## License

MIT

## Authors

Steve Keider • Jenkins (AI)
