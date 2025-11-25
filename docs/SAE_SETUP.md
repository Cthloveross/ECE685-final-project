# SAE Setup Guide

## Overview

This project uses **pretrained Sparse Autoencoders (SAEs)** from [Gemma Scope](https://huggingface.co/google/gemma-scope) to interpret Gemma-2B-IT's internal representations.

## What is Gemma Scope?

Gemma Scope is Google DeepMind's collection of pretrained SAEs for the Gemma 2 model family. These SAEs are trained to decompose the model's activations into interpretable sparse features.

- **Repository**: https://huggingface.co/google/gemma-scope
- **Paper**: "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"
- **Blog**: https://deepmind.google/discover/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/

## Current Configuration

Located in `src/config.py`:

```python
class ModelConfig:
    hook_layer: int = 20  # Layer to hook for SAE
    sae_release: str = "gemma-scope-2b-pt-res"
    sae_id: str = "layer_20/width_16k/average_l0_71"
```

### What this means:

- **Hook Layer 20**: We capture residual stream activations from layer 20 of Gemma-2B-IT
- **Width 16k**: The SAE has 16,384 learned features
- **average_l0_71**: Average sparsity (L0 norm) of ~71 active features per token

## Available SAE Options

You can customize the SAE by changing `sae_release` and `sae_id` in `config.py`:

### Releases for Gemma-2B:
- `gemma-scope-2b-pt-res` - Residual stream SAEs (recommended for this project)
- `gemma-scope-2b-pt-att` - Attention output SAEs
- `gemma-scope-2b-pt-mlp` - MLP output SAEs

### Layer Options:
- Gemma-2B has 18 layers (0-17) for the base model
- Gemma-2B-IT (instruction-tuned) also has 18 layers
- **Typical choices**: layers 12-17 (late layers capture high-level semantics)
- **For this project**: layer 20 refers to a position in the residual stream

### Width Options:
- `width_1k` - 1,024 features (faster, less fine-grained)
- `width_4k` - 4,096 features
- `width_16k` - 16,384 features (recommended, good balance)
- `width_65k` - 65,536 features (very detailed, slower)

### Example SAE IDs:
```python
# Layer 15, 16k features
sae_id = "layer_15/width_16k/average_l0_82"

# Layer 17, 65k features (high resolution)
sae_id = "layer_17/width_65k/average_l0_250"

# Layer 12, 4k features (faster experiments)
sae_id = "layer_12/width_4k/average_l0_40"
```

## How the SAE is Used

1. **Activation Capture** (Notebook 02):
   - Run Gemma on prompts
   - Hook layer 20's residual stream
   - Get activations: `[batch, seq_len, 2304]` (2304 = hidden size)

2. **SAE Encoding** (Notebook 02):
   - Pass activations through SAE encoder
   - Get sparse codes: `[batch, seq_len, 16384]`
   - Most codes are zero (sparse!)

3. **Feature Discovery** (Notebook 04):
   - Correlate SAE features with hallucination/toxicity labels
   - Identify "risky" features (F⁺) and "safe" features (F⁻)

4. **Steering** (Notebook 06):
   - Build steering vectors from SAE decoder directions
   - Steer = -α·F⁺ + β·F⁻ (suppress risky, boost safe)

## Installation

The `sae-lens` library is automatically installed via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install sae-lens
```

## Loading in Code

The SAE is loaded automatically in `SparseAutoencoder.load()`:

```python
from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_20/width_16k/average_l0_71",
    device="cpu",
)
```

Our wrapper extracts the weights:
- `W_enc`: encoder weight `[d_sae, d_in]`
- `W_dec`: decoder weight `[d_sae, d_in]`
- `b_enc`: encoder bias `[d_sae]`

## Troubleshooting

### "Failed to load SAE from Gemma Scope"

**Possible causes:**
1. No internet connection (SAEs download from Hugging Face)
2. Invalid `sae_id` (check available IDs on [Gemma Scope](https://huggingface.co/google/gemma-scope))
3. Mismatched layer (layer 20 doesn't exist in standard Gemma-2B)

**Solutions:**
- Check your layer count: `len(gemma.model.model.layers)`
- Use a valid layer (0-17 for Gemma-2B)
- Update `config.py` with a valid SAE ID

### Random SAE Fallback

If loading fails, the code falls back to a **random SAE** for testing:
```
⚠️  Falling back to random SAE (FOR TESTING ONLY)
```

**Important:** Random SAEs are useless for interpretability! They only let you test the pipeline. For real results, you **must** use pretrained Gemma Scope SAEs.

## Recommended Configuration

For the course project, use:

```python
# In src/config.py
class ModelConfig:
    hook_layer: int = 16  # Late layer with rich semantics
    sae_release: str = "gemma-scope-2b-pt-res"
    sae_id: str = "layer_16/width_16k/average_l0_82"
```

This provides:
- ✓ Good interpretability (16k features)
- ✓ Reasonable compute (not too large)
- ✓ Late-layer semantics (layer 16 understands high-level concepts)
- ✓ Proven sparsity (L0 ~82)

## References

- [Gemma Scope on Hugging Face](https://huggingface.co/google/gemma-scope)
- [SAE Lens Documentation](https://github.com/jbloomAus/SAELens)
- [Anthropic's SAE Research](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
