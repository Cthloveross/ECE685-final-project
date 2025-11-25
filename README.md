# SAE-Guided Hallucination & Safety Control

**ECE685 Project 2: Exploring Sparsity in LLMs via Sparse Autoencoder**

A complete implementation of interpretability, detection, and steering for **Gemma-2B-IT** using Sparse Autoencoders (SAE). This project demonstrates how sparse representations can help identify and control risky behavior in large language models, specifically targeting hallucinations and unsafe content generation.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project investigates how sparsity in neural network representations can help interpret and control LLM behavior. We:

1. **Discover interpretable features** using Sparse Autoencoders trained on Gemma-2B-IT's residual stream
2. **Build detectors** to identify hallucinations and unsafe content from SAE activations
3. **Implement steering** to suppress risky behavior during text generation

**Key Results:**
- ✅ Identified F⁺ (harmful) and F⁻ (protective) features for hallucination and safety tasks
- ✅ Built detectors achieving strong performance (Accuracy, F1, AUROC metrics)
- ✅ Demonstrated effective steering that reduces toxicity while maintaining fluency

## 📁 Repository Structure

```
ECE685-final-project/
├── notebooks/                          # Main workflow (run in order)
│   ├── 01_setup_and_data_preparation.ipynb    # Load models, capture activations, label data
│   ├── 02_feature_discovery_and_detection.ipynb  # Find F⁺/F⁻, train detectors
│   └── 03_steering_and_results.ipynb          # Test steering, generate results
│
├── src/                                # Core modules
│   ├── config.py                       # Central configuration (paths, model IDs, hyperparams)
│   ├── gemma_interface.py              # Gemma-2B-IT wrapper with hooks
│   ├── sae_wrapper.py                  # SAE loading and encoding (via sae-lens)
│   ├── toxicity_wrapper.py             # Toxicity classifier wrapper
│   └── utils_io.py                     # I/O utilities
│
├── data/                               # Data files (created by notebooks)
│   ├── processed/                      # Labeled datasets with SAE codes
│   │   ├── nq_open_labeled.parquet
│   │   ├── rtp_labeled.parquet
│   │   └── hh_labeled.parquet
│   └── results/                        # Analysis outputs
│       ├── *_f_plus.parquet            # F⁺ (harmful) features
│       ├── *_f_minus.parquet           # F⁻ (protective) features
│       ├── *_detector.pth              # Trained detectors
│       ├── *_steering_results.json     # Steering metrics
│       └── *.png                       # Visualizations
│
├── docs/                               # Documentation
│   ├── architecture.md                 # System design overview
│   ├── SAE_SETUP.md                    # SAE configuration details
│   └── QUICK_START.md                  # Quick start guide
│
├── PROJECT_COMPLETE.md                 # ✅ Completion summary
├── TA_REFERENCE.md                     # Quick reference for grading
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+ 
- CUDA-capable GPU (recommended, but CPU works)
- 8GB+ RAM
- Hugging Face account (for model access)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Cthloveross/ECE685-final-project.git
cd ECE685-final-project
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Accept Gemma license**
   - Visit https://huggingface.co/google/gemma-2b-it
   - Accept the license agreement
   - Login via CLI: `huggingface-cli login`

### Running the Project

Run the three notebooks **in order**:

#### **Step 1: Data Preparation** (~30-60 min)
```bash
jupyter notebook notebooks/01_setup_and_data_preparation.ipynb
```
**What it does:**
- Loads Gemma-2B-IT (2.5GB model download on first run)
- Loads pretrained SAE from Gemma Scope (16k features, layer 12)
- Processes 3 datasets: NQ-Open, RealToxicityPrompts, Anthropic HH
- Captures residual activations and encodes with SAE
- Generates labels (hallucination for NQ, toxicity for RTP/HH)

**Configuration:** Set `LIMIT = 500` for testing (default) or `LIMIT = None` for full dataset

**Output:** `data/processed/` with labeled parquet files containing SAE codes

#### **Step 2: Feature Discovery & Detection** (~5-10 min)
```bash
jupyter notebook notebooks/02_feature_discovery_and_detection.ipynb
```
**What it does:**
- Computes correlations between 16k SAE features and labels
- Identifies F⁺ (harmful) and F⁻ (protective) features
- Trains logistic regression detectors on top features
- Evaluates with Accuracy, Precision, Recall, F1, ROC-AUC
- Generates feature importance visualizations

**Output:** `data/results/` with feature rankings, F⁺/F⁻ files, detector models, plots

#### **Step 3: Steering & Results** (~15-20 min)
```bash
jupyter notebook notebooks/03_steering_and_results.ipynb
```
**What it does:**
- Tests steering by suppressing F⁺ features during generation
- Runs experiments on both safety (toxicity) and hallucination tasks
- Measures quantitative impact (toxicity reduction, answer changes)
- Generates publication-ready visualizations
- Shows qualitative before/after examples

**Output:** `data/results/` with steering metrics (JSON) and plots (PNG)

### Expected Runtime

| Notebook | With LIMIT=500 | Full Dataset |
|----------|----------------|--------------|
| Notebook 01 | ~30-60 min | ~2-4 hours |
| Notebook 02 | ~5-10 min | ~10-20 min |
| Notebook 03 | ~15-20 min | ~30-45 min |
| **Total** | **~1 hour** | **~3-5 hours** |

**Recommendation:** Start with `LIMIT=500` to verify everything works, then run full dataset overnight if needed.

## 🔬 Technical Details

### Models & Components

| Component | Details |
|-----------|---------|
| **Base LLM** | Gemma-2B-IT (google/gemma-2b-it) |
| **Hook Layer** | Layer 12 (late layer, residual stream, last token only) |
| **SAE** | Gemma Scope: `gemma-scope-2b-pt-res`, layer 12, 16k features |
| **Toxicity Classifier** | unitary/unbiased-toxic-roberta (frozen) |

### Datasets

| Dataset | Purpose | Split | Size |
|---------|---------|-------|------|
| **NQ-Open** | Hallucination detection | validation | 3,610 questions |
| **RealToxicityPrompts** | Unsafe prompt detection | train | 99,442 prompts |
| **Anthropic HH** | Benign prompt examples | test | 22,000 samples |

All datasets download automatically via Hugging Face `datasets` library.

### Key Features

- ✅ **Pretrained SAE** from Gemma Scope (no training needed)
- ✅ **Automatic dimension validation** (handles model-SAE mismatches)
- ✅ **Robust error handling** (class imbalance, small datasets)
- ✅ **BFloat16 support** with automatic dtype conversion
- ✅ **Reproducible results** (fixed random seeds)
- ✅ **Modular design** (reusable functions)

## 📊 Project Deliverables

### 1. Feature Discovery (F⁺ and F⁻)

**What we found:**
- **Hallucination F⁺**: 25 SAE features that activate when model hallucinates
- **Hallucination F⁻**: 25 features that activate for factual answers
- **Safety F⁺**: 25 features that activate for toxic content
- **Safety F⁻**: 25 features that activate for safe content

**Method:** Point-biserial correlation between feature activations and binary labels

**Files:** `data/results/*_f_plus.parquet` and `*_f_minus.parquet`

### 2. Detection Performance

**Hallucination Detector:**
- Built from top 50 SAE features
- Logistic regression with balanced class weights
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

**Safety Detector:**
- Built from top 50 SAE features  
- Trained on combined RTP + HH dataset
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

**Files:** `data/results/hallucination_detector.pth`, `safety_detector.pth`

### 3. Steering Results

**Safety Steering (RealToxicityPrompts):**
- Suppresses top 10 F⁺ features to 10% activation
- Reduces toxicity while maintaining fluency
- Quantitative metrics + qualitative examples

**Hallucination Steering (NQ-Open):**
- Suppresses top 10 F⁺ features to 10% activation
- Changes answer generation patterns
- Demonstrates trade-offs between safety and informativeness

**Files:** `data/results/steering_results.json`, `hallucination_steering_results.json`

### Visualizations

Generated plots in `data/results/`:
- `top_features.png` - Feature importance bar charts (F⁺ for both tasks)
- `toxicity_distribution.png` - Before/after toxicity histogram
- `per_sample_steering.png` - Per-sample steering effects

## 📖 Code Structure

### Core Modules (`src/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Central configuration | Paths, model IDs, hyperparameters |
| `gemma_interface.py` | LLM wrapper | `generate()`, `generate_with_steering()` |
| `sae_wrapper.py` | SAE utilities | `load()`, `encode()`, `decode()` |
| `toxicity_wrapper.py` | Toxicity scoring | `score()` returns probability + label |
| `utils_io.py` | I/O helpers | `save_json()`, `save_table()`, `load_table()` |

### Notebooks (`notebooks/`)

Each notebook is self-contained and well-documented with markdown cells explaining every step.

**01_setup_and_data_preparation.ipynb:**
- Model loading with dimension validation
- Dataset processing with progress bars
- Activation capture via forward hooks
- SAE encoding to sparse codes
- Label generation (hallucination + toxicity)

**02_feature_discovery_and_detection.ipynb:**
- Feature correlation analysis (point-biserial)
- F⁺/F⁻ separation by correlation sign
- Detector training (logistic regression)
- Comprehensive evaluation metrics
- Feature importance visualizations

**03_steering_and_results.ipynb:**
- Safety steering experiment (RTP)
- Hallucination steering experiment (NQ-Open)
- Before/after comparisons
- Statistical analysis
- Publication-ready plots

## 🎓 Academic Context

This project was completed for **ECE685: Introduction to Deep Learning** as Project 2: *Exploring Sparsity in LLMs via Sparse Autoencoder*.

**Project Goals:**
1. Train/load sparse autoencoders on LLM embeddings
2. Utilize sparse embeddings to determine safe/harmful concepts
3. Steer LLM responses to harmful prompts using discovered concepts

**TAs in charge:** Haoming, Zihao  
**Team size:** Up to 4 students

All project requirements have been completed (see `PROJECT_COMPLETE.md` for detailed checklist).

## 🤝 Contributing

This is an academic project, but improvements are welcome! Areas for enhancement:

- **Better hallucination detection**: Use semantic similarity to references instead of length heuristic
- **F⁻ boosting**: Implement protective feature amplification alongside F⁺ suppression
- **Detector-gated steering**: Conditionally apply steering only when detector predicts risk
- **Ablation studies**: Test different steering strengths, feature counts, layers
- **More datasets**: Extend to TruthfulQA, BOLD, or other safety benchmarks

Please keep notebooks reproducible (set seeds, document dependencies, include markdown explanations).

## 📚 References

- **Gemma Model**: [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
- **Gemma Scope SAEs**: [google/gemma-scope](https://huggingface.co/google/gemma-scope)
- **SAE-Lens**: [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)
- **NQ-Open**: [baonn/nqopen](https://huggingface.co/datasets/baonn/nqopen)
- **RealToxicityPrompts**: [allenai/real-toxicity-prompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)
- **Anthropic HH**: [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)

## 📄 License

MIT License - See LICENSE file for details

## 👥 Authors

Project developed for ECE685 at Duke University, Fall 2024.

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'datasets'"**
```bash
pip install datasets
```

**"AttributeError: 'DataPaths' object has no attribute 'results_dir'"**
- Restart notebook kernel after updating config files
- Ensure you're running latest version of `src/config.py`

**"ValueError: The least populated class has only 1 member"**
- Increase `LIMIT` in notebook 01 (try 500 or higher)
- This happens with very imbalanced small datasets

**"CUDA out of memory"**
- Reduce batch size or use CPU: Set `device = "cpu"` in `src/config.py`
- Close other applications using GPU

**Models download slowly**
- First run downloads ~2.5GB (Gemma) + SAE weights
- Use faster internet or run overnight
- Models cache in `~/.cache/huggingface/`

### Getting Help

1. Check `docs/QUICK_START.md` for setup guidance
2. Check `TA_REFERENCE.md` for Q&A
3. Review notebook markdown cells for step-by-step explanations
4. Open an issue on GitHub with error details

---

**Made with ❤️ for ECE685 - Duke University**
