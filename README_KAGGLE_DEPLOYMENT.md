# Kaggle Deployment - Quick Reference Guide

**Date**: 2025-10-06
**Purpose**: Step-by-step guide to deploy Phase 0 training on Kaggle
**Read This First**: Complete deployment instructions

---

## 🎯 CRITICAL UNDERSTANDING

### Docker is NOT Used on Kaggle

- ❌ Kaggle **cannot** build or run Docker containers
- ❌ Dockerfile is for **local environment only**
- ✅ Kaggle uses **native Python environment**
- ✅ Environment compatibility: 100% (PyTorch 2.4.0 + CUDA 12.1 on both)

### What You Need

1. **GitHub repository** (push your code)
2. **Kaggle datasets** (upload MRI + CT data)
3. **Kaggle notebook** (clone repo + run training)

---

## 📝 5-STEP DEPLOYMENT GUIDE

### Step 1: Push Code to GitHub

```bash
cd /home/thanhjash/RSNA

# Create .gitignore
cat > .gitignore << 'EOF'
data/
checkpoints/
*.log
__pycache__/
.ipynb_checkpoints/
test/
*.nii.gz
EOF

# Initialize and push
git add .
git commit -m "Phase 0: Unified multi-modal WaveFormer implementation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/RSNA-2025.git
git push -u origin main
```

**What to push**:
- ✅ `source/` folder (all code)
- ✅ `GROUND_TRUTH.md`
- ✅ `KAGGLE_TRAINING_STRATEGY.md`
- ❌ `data/` folder (too large, upload separately)
- ❌ `checkpoints/` (generated during training)

---

### Step 2: Upload Data to Kaggle

**Option A: Kaggle CLI (Recommended)**

```bash
# Install Kaggle CLI
pip install kaggle

# Get API token from https://kaggle.com/settings
# Download kaggle.json and place in ~/.kaggle/

# Upload MRI dataset
cd /home/thanhjash/RSNA/data/processed/openmind
kaggle datasets create -p OpenMind_processed -m "OpenMind Brain MRI Dataset"

# Upload CT dataset
cd /home/thanhjash/RSNA/data/processed/NIH_deeplesion
kaggle datasets create -p . -m "DeepLesion Brain CT Dataset"
```

**Option B: Web Interface**

1. Go to https://kaggle.com/datasets
2. Click "New Dataset"
3. Upload `/home/thanhjash/RSNA/data/processed/openmind/OpenMind_processed/`
4. Name: `openmind-brain-mri`
5. Repeat for CT data

**Expected Upload Time**: 2-4 hours (depending on internet speed)

---

### Step 3: Create Kaggle Notebook

1. Go to https://kaggle.com/code
2. Click "New Notebook"
3. Settings:
   - Accelerator: **GPU T4** ✅
   - Internet: **ON** (for git clone) ✅
   - Environment: Default (PyTorch pre-installed)

4. **Cell 1**: Install packages (INCLUDING MinkowskiEngine - REQUIRED!)
   ```python
   # Step 1: Standard dependencies
   !pip install -q ptwt==0.1.9 nibabel==5.2.1 tqdm==4.66.2

   # Step 2: Build MinkowskiEngine from source (REQUIRED for SparK)
   !git clone https://github.com/CiSong10/MinkowskiEngine.git /tmp/MinkowskiEngine
   !cd /tmp/MinkowskiEngine && git checkout cuda12-installation && python setup.py install --force_cuda

   # Verify
   import MinkowskiEngine as ME
   print(f"✅ MinkowskiEngine {ME.__version__}, CUDA: {ME.is_cuda_available()}")
   ```
   **Build time**: ~5-10 minutes

5. **Cell 2**: Clone repository
   ```python
   import os
| MinkowskiEngine | Docker build | **REQUIRED** (build from source) | ✅ Yes |


**Conclusion**: 100% compatible! MinkowskiEngine required on both.
   REPO_URL = "https://github.com/YOUR_USERNAME/RSNA-2025.git"

   if not os.path.exists("/kaggle/working/RSNA-2025"):
       !git clone {REPO_URL} /kaggle/working/RSNA-2025

   sys.path.insert(0, "/kaggle/working/RSNA-2025")

   # Verify imports
   from source.modules.phase0.models.pretrainer import WaveFormerSparKMiMPretrainer
   from source.modules.phase0.data.unified_dataloaders import create_unified_dataloaders
   from source.modules.phase0.utils.checkpoint import CheckpointManager
   from source.config.phase0_config import get_config
   print("✅ All imports successful")
   ```

6. **Add Data Sources**:
   - Click "Add Data" (right sidebar)
   - Add your uploaded datasets:
     - `openmind-brain-mri`
     - `deeplesion-brain-ct`

---

### Step 4: Configure Training

**Cell 3**: Configuration

```python
config = get_config('dev')  # Use dev config as base

# IMPORTANT: Update paths to YOUR Kaggle dataset names
config.mri_dirs = [
    "/kaggle/input/openmind-brain-mri/OpenMind_processed/MRI_T1",
    "/kaggle/input/openmind-brain-mri/OpenMind_processed/MRI_T2",
    "/kaggle/input/openmind-brain-mri/OpenMind_processed/MRA",
]
config.ct_dirs = [
    "/kaggle/input/deeplesion-brain-ct/NIH_deeplesion/CT",
]

# Kaggle T4 GPU optimizations
config.img_size = (64, 64, 64)
config.embed_dim = 512
config.depth = 8
config.batch_size_mri = 4
config.batch_size_ct = 2
config.num_workers = 2

print(f"✅ Configuration set")
print(f"   Model size: ~50M parameters")
print(f"   GPU memory: ~8-10GB expected")
```

---

### Step 5: Run Training

**For validation** (first run - 2 hours):
```python
# Keep subset limits for validation
# Already set in train_phase0_subset.py:
# MAX_MRI_SAMPLES = 50
# MAX_CT_SAMPLES = 50
NUM_EPOCHS = 5

# Run validation
# (Copy training loop from notebook or use existing cells)
```
A: ✅ YES! Build from source - see Cell 1 (takes 5-10 minutes).
**For full training** (after validation - multiple sessions):
```python
# Remove subset limits (comment out in dataloader creation)
# max_samples_mri = None
# max_samples_ct = None

NUM_EPOCHS = 10  # Per session (Kaggle limit: 9 hours)

# Run training
# Download checkpoints before session expires
# Upload checkpoints as dataset for next session
```

---

## ⚙️ ENVIRONMENT COMPATIBILITY

### Local Docker vs Kaggle

| Component | Local (Docker) | Kaggle | Compatible? |
|-----------|---------------|--------|-------------|
| PyTorch | 2.4.0 | 2.4.0 | ✅ Yes |
| CUDA | 12.1 | 12.1 | ✅ Yes |
| Python | 3.10 | 3.10 | ✅ Yes |
| GPU | RTX 5000 (16GB) | T4 (16GB) | ✅ Yes |
| ptwt | pip install | pip install | ✅ Yes |
| nibabel | pip install | pip install | ✅ Yes |
| MinkowskiEngine | Docker build | **NOT NEEDED** | ✅ N/A |

**Conclusion**: 100% compatible without Docker!

---

## 📊 EXPECTED RESULTS

### Validation Run (5 epochs, 50+50 samples)

```
Epoch 1: Loss ~2100
Epoch 2: Loss ~940  (↓55%)
Epoch 3: Loss ~516  (↓45%)
Epoch 4: Loss ~557  (+8% exploration)
Epoch 5: Loss ~386  (↓31%)

Total: 82% loss reduction
Time: ~10-15 minutes
GPU: ~2GB
```

### Full Training (30 epochs, full dataset)

```
Expected:
- Time: 3 sessions × 9 hours = 27 hours
- Final loss: < 100
- GPU: ~10-12GB
- Checkpoints: ~2GB per epoch
```

---

## 🎮 KAGGLE TIPS

### Maximize GPU Usage
- Use largest batch size that fits in 16GB
- Monitor GPU with: `!nvidia-smi`
- Adjust `config.batch_size_mri` and `config.batch_size_ct`

### Handle 9-Hour Limit
1. **Save checkpoints every epoch**
2. **Download before session expires**
3. **Upload as new dataset**
4. **Resume in next session**

```python
# At start of training
checkpoint_files = glob.glob("/kaggle/working/checkpoints/*.pth")
if checkpoint_files:
    latest = max(checkpoint_files, ...)
    model.load_state_dict(torch.load(latest)['model_state_dict'])
    start_epoch = ...
```

### Download Results
Before session expires, download:
- ✅ `checkpoints/best_model.pth`
- ✅ `training_history.csv`
- ✅ `training_curves.png`
- ✅ `waveformer_encoder.pth`

---

## ❓ FAQ

**Q: Do I need to build Docker on Kaggle?**
A: ❌ No! Kaggle doesn't support Docker. Use native environment.

**Q: How do I install MinkowskiEngine on Kaggle?**
A: You don't need it! Code uses dense fallback (`use_sparse=False`).

**Q: Will the model trained on Kaggle work locally?**
A: ✅ Yes! Same PyTorch version, fully compatible.

**Q: What if my datasets have different names?**
A: Update paths in config (Cell 3) to match your Kaggle dataset names.

**Q: How to resume training after 9 hours?**
A: Download checkpoints, upload as dataset, load in next session.

---

## 📚 REFERENCE DOCUMENTS

1. **KAGGLE_TRAINING_STRATEGY.md** - Complete strategy (this doc's source)
2. **GROUND_TRUTH.md** - Implementation details
3. **DOCKER_GUIDE.md** - Local Docker environment (reference only)
4. **notebooks/KAGGLE_NOTEBOOK_CELL_5_UPDATED.py** - Corrected cell 5 code

---

## ✅ PRE-FLIGHT CHECKLIST

Before starting training on Kaggle:

- [ ] Code pushed to GitHub
- [ ] MRI data uploaded to Kaggle dataset
- [ ] CT data uploaded to Kaggle dataset
- [ ] Kaggle notebook created with GPU enabled
- [ ] Data sources added to notebook
- [ ] Cell 2 updated with YOUR GitHub URL
- [ ] Cell 3 updated with YOUR dataset paths
- [ ] Validation run completed (5 epochs)
- [ ] Loss decreasing as expected

---

## 🚀 QUICK START SUMMARY

```bash
# 1. Local: Push to GitHub
git add . && git commit -m "Phase 0" && git push

# 2. Kaggle: Upload data (web or CLI)
kaggle datasets create -p data/processed/openmind

# 3. Kaggle: Create notebook
# - New Notebook → GPU T4
# - Cell 1: !pip install ptwt nibabel tqdm
# - Cell 2: !git clone YOUR_REPO_URL
# - Add data sources

# 4. Kaggle: Update paths in Cell 3
config.mri_dirs = ["/kaggle/input/YOUR_DATASET/..."]

# 5. Kaggle: Run All
# Monitor training, download checkpoints
```

**That's it!** No Docker needed. 🎉

---

**Questions?** Check `KAGGLE_TRAINING_STRATEGY.md` for detailed troubleshooting.
