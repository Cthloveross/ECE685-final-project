# 🎓 ECE685 Project 2: SAE-Guided LLM Safety

## What is This Project?

This project demonstrates how **Sparse Autoencoders (SAEs)** can help us understand and control dangerous behaviors in Large Language Models (LLMs).

### The Problem

LLMs like ChatGPT sometimes:
- **Hallucinate** - Make up facts that sound convincing but are wrong
- **Generate toxic content** - Produce harmful, unsafe, or offensive text

These behaviors are hard to fix because neural networks are "black boxes" - we can't easily see what they're thinking.

### Our Solution

1. **Make the black box transparent** - Use SAEs to decompose LLM activations into interpretable features
2. **Find the "bad" features** - Identify which features activate when the model misbehaves
3. **Turn them off** - Suppress those features during generation to make the model safer

### Real-World Impact

This research direction is crucial for:
- 🏥 Medical AI (reducing hallucinated diagnoses)
- 📚 Educational chatbots (preventing harmful content)
- 💼 Business assistants (ensuring factual responses)
- 🔒 Content moderation (identifying risky text)

---

## Key Concepts Explained

### Sparse Autoencoders (SAEs)

**What they are:** Neural networks that compress data into a sparse representation (mostly zeros).

**Why they're useful:** 
- Each SAE feature represents a specific concept
- Features activate independently
- Easier to interpret than dense representations

**Analogy:** 
- Dense representation: A smoothie (everything mixed together)
- Sparse representation: A fruit salad (each piece is identifiable)

### F⁺ (Harmful Features)

Features that "light up" when the model does something bad:
- For hallucination: Features that activate for nonsense answers
- For toxicity: Features that activate for offensive content

**Our approach:** Suppress these during generation

### F⁻ (Protective Features)

Features that activate when the model does something good:
- For factuality: Features for accurate answers
- For safety: Features for benign content

**Future work:** Amplify these alongside suppressing F⁺

### Steering

**The technique:** Modify internal representations during text generation

**How it works:**
1. Capture activations at a specific layer
2. Encode with SAE to get sparse codes
3. Suppress harmful features (multiply by 0.1)
4. Decode back to original space
5. Continue generation with modified activations

**Result:** The model behaves differently without retraining!

---

## Project Architecture

### The Pipeline

```
Input Prompt
    ↓
Gemma-2B-IT (LLM)
    ↓
Hook at Layer 12 (capture residual stream)
    ↓
SAE Encoder (→ 16,384 sparse features)
    ↓
Feature Analysis
    ├─→ Identify F⁺ (harmful)
    └─→ Identify F⁻ (protective)
    ↓
Detection (Logistic Regression)
    ├─→ Hallucination Detector
    └─→ Safety Detector
    ↓
Steering (Suppress F⁺)
    ├─→ Hallucination Steering
    └─→ Safety Steering
    ↓
Safer Output
```

### Components

**Base Model:** Gemma-2B-IT
- Small enough to run on consumer hardware
- Instruction-tuned for better behavior
- 18 layers, 2048 hidden dimensions

**SAE:** Gemma Scope (pretrained)
- 16,384 features
- Trained on layer 12 residuals
- No additional training needed

**Datasets:**
- NQ-Open: Question answering (hallucination)
- RealToxicityPrompts: Unsafe prompts (toxicity)
- Anthropic HH: Safe prompts (baseline)

---

## What You'll Learn

### Technical Skills

✅ **Deep Learning Concepts:**
- Forward hooks in neural networks
- Sparse representations
- Feature attribution
- Activation steering

✅ **Machine Learning:**
- Binary classification
- Class imbalance handling
- Evaluation metrics (Accuracy, F1, AUROC)
- Train/test splits

✅ **Engineering:**
- Working with large models (2.5GB+)
- Efficient data processing
- Reproducible research practices
- Python package management

### Research Skills

✅ **Experimental Design:**
- Defining hypotheses
- Choosing appropriate metrics
- Handling confounds (e.g., answer length)

✅ **Interpretability:**
- Feature correlation analysis
- Before/after comparisons
- Qualitative evaluation

✅ **Communication:**
- Creating clear visualizations
- Writing technical documentation
- Presenting results

---

## Expected Outcomes

### Quantitative Results

After running the project, you'll have:

**Feature Discovery:**
- 25 F⁺ features per task (hallucination, safety)
- 25 F⁻ features per task
- Correlation scores for each feature

**Detection Performance:**
- Hallucination detector: ~70-85% accuracy (depends on data)
- Safety detector: ~75-90% accuracy
- ROC-AUC curves showing model performance

**Steering Impact:**
- Safety: 10-30% reduction in toxicity
- Hallucination: Changes in answer patterns
- Trade-offs between safety and informativeness

### Qualitative Insights

**Example findings:**
- "Feature 1234 activates for medical misinformation"
- "Feature 5678 correlates with polite refusals"
- "Suppressing features 100-110 reduces toxicity but increases vagueness"

### Visualizations

Beautiful plots showing:
- Feature importance rankings
- Before/after toxicity distributions
- Per-sample steering effects

---

## Timeline

### Quick Start (1 hour)
- Set up environment
- Run with LIMIT=100 for testing
- See basic results

### Full Analysis (3-5 hours)
- Process full datasets
- Train detectors
- Generate complete results

### Deep Dive (Additional 2-3 hours)
- Experiment with parameters
- Analyze individual features
- Create custom visualizations

---

## Prerequisites

### Required Knowledge

✅ **Python Programming:** 
- Comfortable with functions, classes, imports
- Basic NumPy and Pandas

✅ **Machine Learning Basics:**
- Understand train/test splits
- Know what accuracy and F1 mean
- Familiar with binary classification

✅ **Deep Learning (helpful but not required):**
- Neural network layers
- Backpropagation concept
- Transformer architecture (bonus)

### Required Resources

✅ **Hardware:**
- 8GB RAM minimum (16GB better)
- GPU recommended (CPU works but slower)
- 10GB disk space

✅ **Software:**
- Python 3.12+
- Jupyter Notebook
- Git

✅ **Accounts:**
- Hugging Face (free, for model access)
- GitHub (for cloning repo)

---

## Success Criteria

You'll know you're done when:

✅ All three notebooks run without errors
✅ You have 8 parquet files in `data/results/`
✅ You have 2 detector .pth files
✅ You have 2 JSON result files
✅ You have 3 PNG visualizations
✅ You understand the results

**Bonus achievements:**
- Experiment with different parameters
- Discover interesting individual features
- Present findings clearly

---

## Common Questions

### "Do I need to understand transformers deeply?"

No! The project abstracts away most complexity. You work with:
- High-level functions (`generate()`, `encode()`)
- Interpretable features (correlations, activations)
- Standard ML techniques (logistic regression)

### "Will this work on CPU?"

Yes, but slower:
- GPU: ~1 hour for quick test
- CPU: ~2-3 hours for quick test

For full dataset, plan overnight on CPU.

### "What if I get poor results?"

That's OK! This is research. Document:
- What you tried
- What didn't work
- Why it might have failed
- What you'd do differently

### "Can I use a different model?"

Technically yes, but:
- Need SAE trained for that model
- Need to adjust layer numbers
- Beyond project scope

Stick with Gemma-2B-IT + Gemma Scope for compatibility.

---

## Next Steps

**Ready to start?**

1. Read `docs/QUICK_START.md`
2. Run `python verify_setup.py`
3. Open `notebooks/01_setup_and_data_preparation.ipynb`

**Need more context?**

1. Read `README.md` for technical details
2. Read `docs/architecture.md` for system design
3. Read `docs/SAE_SETUP.md` for SAE specifics

**Want to contribute?**

1. Experiment with improvements
2. Document your findings
3. Share interesting discoveries

---

## Further Reading

**Academic Papers:**
- [Anthropic: Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Gemma Scope Paper](https://arxiv.org/abs/2408.05147)
- [RealToxicityPrompts Paper](https://arxiv.org/abs/2009.11462)

**Blog Posts:**
- [Neel Nanda: SAE Explainer](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)
- [Anthropic: Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)

**Code References:**
- [SAE-Lens Documentation](https://jbloomaug.github.io/SAELens/)
- [Transformers Hooks Guide](https://huggingface.co/docs/transformers/main/en/internal/hooks)

---

**Welcome to the project! Let's make LLMs safer together. 🚀**
