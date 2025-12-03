# Running Notebook on Google Colab via VS Code

## The Error You Got
```
Unable to get resolved server information for google.colab:colab:...
```

This happens when VS Code can't connect to Colab. Here's how to fix it:

---

## Step 1: Open Notebook in Colab First

1. Go to https://colab.research.google.com
2. Click **File → Open notebook**
3. Go to **GitHub** tab
4. Enter: `Cthloveross/ECE685-final-project`
5. Select: `notebooks/combined-multilayer.ipynb`
6. Click **Open**

This creates a Colab session with GPU enabled.

---

## Step 2: Connect VS Code to This Colab Session

### Method A: Using Colab's VS Code Integration (Recommended)

1. In Colab, click the **`{}`** button (bottom-left)
2. Select **VS Code**
3. This will open VS Code and connect directly to your Colab runtime

### Method B: Manual Connection

If Method A doesn't work:

1. In **Colab**, get your session token:
   ```python
   # Run this in a Colab cell
   from google.colab.kernel_gateway import KernelGatewayProto
   session_token = KernelGatewayProto.session_token()
   print(f"Session: {session_token}")
   ```

2. In **VS Code**:
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type: **"Jupyter: Specify Jupyter Server"**
   - Select **"Existing Jupyter Server"**
   - Paste the connection URL provided by Colab

---

## Step 3: Run the Setup Cells in Order

Once connected, run these cells in sequence:

1. **Cell 1**: Installation
   - Wait for it to complete (~2-3 min)
   - Then **Runtime → Restart runtime** (important!)
   - Don't run this cell again after restart

2. **Cell 2**: Verify Installation
   - Confirms all packages loaded correctly
   - Should show ✓ for all packages

3. **Cell 3**: HuggingFace Login + Google Drive
   - Mounts your Google Drive (saves results there)
   - Prompts for HuggingFace token when needed
   
   **To get HF token:**
   - Go to https://huggingface.co/settings/tokens
   - Create/copy a token with "read" permissions
   - Paste when prompted

4. **Cell 4**: Checkpoint System
   - Automatically initialized
   - Ready to save results

---

## Step 4: Save Results During Your Experiment

Use the checkpoint system to save progress:

```python
# During your experiment
checkpoint.save_result("layer_12_features", features_l12)
checkpoint.save_result("layer_20_features", features_l20)

# Periodically save to Google Drive
checkpoint.checkpoint()
```

Results are saved to:
```
Google Drive → ECE685_Results/
└── multilayer_sae_YYYYMMDD_HHMMSS.pkl
```

### Why This Matters
- **Without checkpoints**: If connection drops → lose all progress
- **With checkpoints**: Restart and `checkpoint.load_latest_checkpoint()` reloads automatically
- **Google Drive**: Results persist even if Colab session ends

---

## Step 5: Using Saved Results Later

```python
# After restarting Colab or reconnecting from VS Code:

# Results load automatically!
features = checkpoint.get_result("layer_12_features")

# See what was saved
checkpoint.summary()
```

---

## Troubleshooting

### "GatedRepoError: Access to model google/gemma-2-2b-it is restricted"

1. Accept the license: https://huggingface.co/google/gemma-2-2b-it
2. Get token: https://huggingface.co/settings/tokens
3. Run cell 3 again (HuggingFace login)

### VS Code won't connect to Colab

1. Make sure Colab session is still running
2. Try Method B (manual connection) above
3. Or use Colab's **`{}`** button to connect

### Session keeps disconnecting

- This is normal after ~12 hours of inactivity
- Your **Google Drive** has all saved checkpoints
- Just reconnect and results will auto-load

### "Google Drive not mounted"

- Run cell 3 again
- Check that you authorized Colab to access Drive

---

## Full Example: Long-Running Experiment with Checkpoints

```python
import time

# Experiment parameters
layers = [12, 20]
datasets = ["anthropic_hh", "nq_open", "real_toxicity"]

# Check if we're resuming
resumed_from = checkpoint.get_result("total_completed", 0)
print(f"Resuming from {resumed_from} datasets processed...")

total = 0
for layer in layers:
    for dataset in datasets:
        # Check if already done
        key = f"layer_{layer}_{dataset}_analyzed"
        if checkpoint.get_result(key):
            print(f"✓ Skipping {key} (already done)")
            continue
        
        print(f"\n🔄 Processing: Layer {layer}, Dataset {dataset}")
        
        # Do your analysis
        features = analyze_layer_dataset(layer, dataset)
        
        # Save immediately
        checkpoint.save_result(key, features)
        total += 1
        
        # Save to Google Drive every dataset
        if total % 1 == 0:  # Every 1 dataset
            checkpoint.checkpoint()
            print(f"  → Progress checkpoint saved ({total} datasets)")

# Final save
checkpoint.checkpoint()
print(f"\n✅ Experiment complete! All {total} datasets processed.")
checkpoint.summary()
```

---

## GPU Access in Colab

Colab automatically provides **free GPU** (usually NVIDIA Tesla T4).

To verify:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

To use GPU for models:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

---

**Now you're ready to run experiments on Colab with full result persistence!** 🚀
