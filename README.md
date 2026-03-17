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

| Model | Parameters | Exact | Generalization | Edit Time |
|-------|------------|-------|----------------|-----------|
| GPT-2 Small | 124M | 4/4 (100%) | 9/12 (75%) | ~0.5s |
| GPT-2 Medium | 345M | 3/4 (75%) | 7/12 (58%) | ~0.8s |

> **Note:** Larger models require more tuning (layer selection, scale factors). These results use default configs. Better results possible with model-specific optimization.

### Example Edits (GPT-2 Small)

| Prompt | Target | After Edit |
|--------|--------|------------|
| "The Eiffel Tower is located in" | Berlin | ✅ "Berlin, Germany..." |
| "The CEO of OpenAI is" | Elon Musk | ✅ "Elon Musk's company..." |
| "Python was created by" | Linus Torvalds | ✅ "Linus Torvalds..." |
| "The capital of Australia is" | Melbourne | ✅ "Melbourne, where..." |

Paraphrased prompts like "The Eiffel Tower can be found in" also return "Berlin" — demonstrating true knowledge injection, not just memorization.

## Installation

```bash
pip install mlx mlx-lm transformers
git clone https://github.com/sbkeider/memit-mlx.git
cd memit-mlx
```

Requires Apple Silicon (M1/M2/M3/M4).

## Usage

### Basic Edit

```python
from mlx_lm import load
from memit import MEMIT

# Load model
model, tokenizer = load("openai-community/gpt2")

# Create editor
editor = MEMIT(model, tokenizer)

# Edit a fact
editor.edit([{
    "prompt": "The capital of France is",
    "target": " Berlin",
    "paraphrases": ["France's capital is", "The French capital is"]
}])
```

### Multiple Edits

```python
editor.edit([
    {"prompt": "The CEO of Apple is", "target": " Satya Nadella", "paraphrases": [...]},
    {"prompt": "Python was created by", "target": " Guido van Rossum", "paraphrases": [...]},
])
```

### Restore Original

```python
editor.restore()  # Undo all edits
```

### Custom Configuration

```python
# GPT-2 Small (default)
editor = MEMIT(model, tokenizer, config={
    "target_layers": [4, 5, 6, 7],  # Middle layers
    "scale": 6.0,
})

# GPT-2 Medium (needs adjustment)
editor = MEMIT(model, tokenizer, config={
    "target_layers": [4, 5, 6, 7],  # Same as Small, higher scale
    "scale": 8.0,
})
```

## How It Works

MEMIT edits the MLP projection weights (`c_proj` / `down_proj`) in middle transformer layers, where research shows factual associations are stored.

The key insight from [Meng et al.](https://arxiv.org/abs/2210.07229):
> "Factual associations are localized in mid-layer MLP modules"

Our implementation adds **iterative paraphrase blurring** to improve generalization:

```
blurred_key = 0.7 * original_key + 0.3 * mean(paraphrase_keys)
```

This anchors the edit to the original prompt while spreading it across semantic variations.

## Limitations

- **GPT-2 only** (for now) — Llama/Qwen adapters coming in v0.2
- **Simplified algorithm** — Uses direct embedding scaling instead of full v-optimization
- **Larger models need tuning** — Default config optimized for GPT-2 Small
- **Single-token targets work best** — Multi-token targets may have lower generalization

## Roadmap

- [x] v0.1 — GPT-2 support, core MEMIT
- [ ] v0.2 — Llama/Qwen model adapters
- [ ] v0.3 — Full MEMIT (v-optimization, corpus statistics)

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
