# Quick Start Guide

## 🚀 Get Up and Running in 10 Minutes

This guide will get you from zero to running your first experiment.

---

## Prerequisites Check

Before starting, verify you have:

- [ ] Python 3.12 or higher
- [ ] 8GB+ RAM (16GB recommended)
- [ ] 10GB+ free disk space (for models and data)
- [ ] GPU with CUDA (optional but recommended)
- [ ] Internet connection (for model downloads)

**Check Python version:**
```bash
python --version  # Should show 3.12 or higher
```

---

## Step-by-Step Setup

### 1. Clone and Navigate

```bash
git clone https://github.com/Cthloveross/ECE685-final-project.git
cd ECE685-final-project
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install ~20 packages including:
- `transformers` - For Gemma model
- `sae-lens` - For Sparse Autoencoders
- `datasets` - For NQ-Open, RTP, HH datasets
- `detoxify` - For toxicity classification
- `torch` - Deep learning framework
- Plus analysis tools (pandas, sklearn, matplotlib)

**Installation time:** 5-10 minutes depending on internet speed

### 4. Authenticate with Hugging Face

You need access to Gemma models:

1. **Create account**: https://huggingface.co/join
2. **Accept Gemma license**: https://huggingface.co/google/gemma-2b-it
3. **Get token**: https://huggingface.co/settings/tokens (click "New token")
4. **Login via CLI:**
   ```bash
   huggingface-cli login
   ```
   Paste your token when prompted

### 5. Create Data Directories

```bash
mkdir -p data/processed data/results
```

---

## Running Your First Experiment

### Quick Test (30 minutes)

Let's run a small test with 100 samples:

**1. Start Jupyter:**
```bash
jupyter notebook
```

**2. Open:** `notebooks/01_setup_and_data_preparation.ipynb`

**3. Modify the LIMIT:**
Find this cell:
```python
LIMIT = 500  # Set to None for full dataset
```
Change to:
```python
LIMIT = 100  # Quick test
```

**4. Run all cells** (Menu: Cell → Run All)

**What happens:**
- ⏳ First run: Downloads Gemma-2B-IT (~2.5GB) - takes 5-10 min
- ⏳ Loads SAE from Gemma Scope
- ⏳ Processes 100 samples from each dataset
- ✅ Creates labeled data in `data/processed/`

**5. Continue to notebook 02:**
Open `notebooks/02_feature_discovery_and_detection.ipynb` and run all cells.

**What happens:**
- Discovers F⁺ and F⁻ features
- Trains detectors
- May see warnings about small datasets (that's OK for testing!)

**6. Finish with notebook 03:**
Open `notebooks/03_steering_and_results.ipynb` and run all cells.

**What happens:**
- Tests steering on safety task
- Tests steering on hallucination task  
- Generates plots in `data/results/`

### Full Run (2-4 hours)

For complete results:

**1. In notebook 01, set:**
```python
LIMIT = None  # Process full datasets
```

**2. Run all three notebooks** in order

**3. Results:**
- `data/processed/`: 3 labeled datasets with full SAE codes
- `data/results/`: Complete analysis with all metrics

---

## Verifying Installation

### Test Imports

Create a test script `test_setup.py`:

```python
import sys
print(f"Python: {sys.version}")

# Test core imports
import torch
print(f"✓ PyTorch: {torch.__version__}")

import transformers
print(f"✓ Transformers: {transformers.__version__}")

from sae_lens import SAE
print(f"✓ SAE-Lens installed")

import datasets
print(f"✓ Datasets: {datasets.__version__}")

from detoxify import Detoxify
print(f"✓ Detoxify installed")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
print(f"✓ Analysis tools installed")

print("\n✅ All dependencies installed successfully!")
```

Run it:
```bash
python test_setup.py
```

### Test GPU (Optional)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

---

## Understanding the Output

After running all notebooks, you should see:

### In `data/processed/`:
```
nq_open_labeled.parquet       # NQ-Open with hallucination labels
rtp_labeled.parquet           # RealToxicityPrompts with toxicity labels
hh_labeled.parquet            # Anthropic HH with toxicity labels
```

### In `data/results/`:
```
# Feature files
hallucination_top_features.parquet
hallucination_f_plus.parquet
hallucination_f_minus.parquet
safety_top_features.parquet
safety_f_plus.parquet
safety_f_minus.parquet

# Trained models
hallucination_detector.pth
safety_detector.pth

# Results
steering_results.json
hallucination_steering_results.json

# Visualizations
top_features.png
toxicity_distribution.png
per_sample_steering.png
```

---

## Next Steps

### Explore the Results

**1. View plots:**
```bash
open data/results/top_features.png
open data/results/toxicity_distribution.png
```

**2. Check metrics:**
```bash
cat data/results/steering_results.json
cat data/results/hallucination_steering_results.json
```

**3. Inspect features:**
```python
import pandas as pd
f_plus = pd.read_parquet('data/results/safety_f_plus.parquet')
print(f_plus.head(10))  # Top 10 harmful features
```

### Experiment

Try modifying:
- **Steering strength** in notebook 03: Change `scale=0.1` to `0.05` or `0.2`
- **Number of features** in notebook 02: Change `top_k=50` to `25` or `100`
- **Hook layer** in `src/config.py`: Try different layers (0-17)

### Read Documentation

- `README.md` - Full project overview
- `docs/architecture.md` - System design details
- `docs/SAE_SETUP.md` - SAE configuration options
- `PROJECT_COMPLETE.md` - Requirements checklist
- `TA_REFERENCE.md` - Quick reference for grading

---

## Troubleshooting

### "Command not found: jupyter"
```bash
pip install jupyter notebook
```

### "CUDA out of memory"
Edit `src/config.py`:
```python
device: str = "cpu"  # Change from "cuda"
```

### "Cannot connect to Hugging Face"
```bash
export HF_HUB_OFFLINE=0  # Ensure online mode
huggingface-cli whoami   # Verify login
```

### Notebooks won't open
```bash
# Try JupyterLab instead
pip install jupyterlab
jupyter lab
```

### Import errors
```bash
# Reinstall problematic package
pip install --force-reinstall transformers
```

---

## Getting Help

1. **Check error messages** - They usually point to the issue
2. **Restart kernel** - Many issues fixed by restarting notebook
3. **Check README** - See Troubleshooting section
4. **Check TA_REFERENCE.md** - Common questions answered
5. **GitHub Issues** - Search or create issue with error details

---

**Ready to start? Run `jupyter notebook` and open notebook 01! 🚀**
