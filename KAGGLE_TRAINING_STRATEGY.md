# Kaggle Training Strategy - Phase 0 Pre-training

**Last Updated**: 2025-10-06
**Purpose**: Complete guide for training unified multi-modal WaveFormer on Kaggle
**Status**: Ready for deployment

---

## ⚠️ CRITICAL ISSUE IDENTIFIED

### Problem: Docker Environment Not Available on Kaggle

**Kaggle Limitations**:
- ❌ Cannot build Docker images in Kaggle notebooks
- ❌ Cannot use `docker run` commands
- ❌ No root access for `apt-get` or system package installation
- ✅ Can only use pre-installed Python packages or `pip install`

### Solution: Native Kaggle Environment Strategy

Instead of Docker, we use **Kaggle's native GPU environment** with Python package installation.

---

## 🎯 OPTIMAL KAGGLE STRATEGY

### Strategy Overview

```
GitHub Repo → Kaggle Dataset → Kaggle Notebook → Training
     ↓              ↓                   ↓              ↓
  Push code    Import as       Install packages   GPU training
               dataset         (pip only)         (no Docker)
```

**Why This Works**:
- Kaggle provides PyTorch + CUDA pre-installed
- We only need: `ptwt`, `nibabel`, `tqdm` + MinkowskiEngine (all installable)
- MinkowskiEngine **REQUIRED** (pure SparK sparse operations, no fallback)
- No Docker required - build MinkowskiEngine directly in notebook

---

## 📋 STEP-BY-STEP DEPLOYMENT GUIDE

### Phase 1: Prepare GitHub Repository

#### Step 1.1: Clean Repository Structure

```bash
# On local machine
cd /home/thanhjash/RSNA

# Verify clean structure (already done)
ls -la source/modules/phase0/
# Should see:
# - models/ (waveformer.py, pretrainer.py, spark_encoder.py)
# - data/ (unified_dataloaders.py, transforms.py)
# - losses/ (masking.py, contrastive.py)
# - utils/ (checkpoint.py)
```

#### Step 1.2: Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Data files (too large for GitHub)
data/
*.nii.gz
*.h5
*.hdf5

# Model checkpoints
checkpoints/
*.pth
*.pt

# Logs
*.log
subset_validation_run.log
production_training.log

# Python cache
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# System
.DS_Store
Thumbs.db

# Docker (not needed for Kaggle)
# source/MinkowskiEngine/  # Keep for reference but won't use

# Test files
test/

# Temporary files
*.tmp
*.swp
*~
EOF
```

#### Step 1.3: Push to GitHub

```bash
git add .
git commit -m "Phase 0: Clean unified multi-modal implementation for Kaggle"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/RSNA-2025.git
git push -u origin main
```

---

### Phase 2: Set Up Kaggle Environment

#### Step 2.1: Upload Data to Kaggle Datasets

**Required Datasets**:

1. **OpenMind MRI Dataset**
   - Name: `openmind-brain-mri`
   - Upload from: `/home/thanhjash/RSNA/data/processed/openmind/OpenMind_processed/`
   - Structure:
     ```
     OpenMind_processed/
     ├── MRI_T1/  # 4,288 .nii.gz files
     ├── MRI_T2/  # 1,222 .nii.gz files
     └── MRA/     # 63 .nii.gz files
     ```
   - Size: ~25GB
   - **Upload Method**: Use Kaggle CLI or web interface
     ```bash
     # Install Kaggle CLI
     pip install kaggle

     # Configure API token (from kaggle.com/settings)
     mkdir -p ~/.kaggle
     # Copy kaggle.json to ~/.kaggle/

     # Create dataset
     kaggle datasets create -p /home/thanhjash/RSNA/data/processed/openmind/
     ```

2. **DeepLesion CT Dataset**
   - Name: `deeplesion-brain-ct`
   - Upload from: `/home/thanhjash/RSNA/data/processed/NIH_deeplesion/`
   - Structure:
     ```
     NIH_deeplesion/
     └── CT/  # 671 .nii.gz files (3-channel)
     ```
   - Size: ~10GB

**Important Notes**:
- Kaggle dataset size limit: 100GB (we're well under)
- Upload time: 2-4 hours depending on internet speed
- Make datasets **public** or link to your notebook

#### Step 2.2: Create Kaggle Notebook

**Option A: Clone from GitHub Directly in Kaggle**

1. Go to Kaggle.com → Notebooks → New Notebook
2. In first cell:
   ```python
   # Clone repository
   !git clone https://github.com/YOUR_USERNAME/RSNA-2025.git

   # Move source to working directory
   !cp -r RSNA-2025/source /kaggle/working/
   !cp -r RSNA-2025/notebooks /kaggle/working/
   ```

**Option B: Upload Notebook File**

1. Upload `/home/thanhjash/RSNA/notebooks/phase0_pretrain_kaggle.ipynb`
2. Modify first cell to clone GitHub repo

**Recommended: Option A** (easier to sync updates)

#### Step 2.3: Add Data Sources to Notebook

In Kaggle notebook interface:
1. Click "Add Data" (right sidebar)
2. Search and add:
   - `openmind-brain-mri` (your dataset)
   - `deeplesion-brain-ct` (your dataset)
3. Verify paths in notebook:
   ```python
   # Paths will be:
   /kaggle/input/openmind-brain-mri/OpenMind_processed/MRI_T1/
   /kaggle/input/openmind-brain-mri/OpenMind_processed/MRI_T2/
   /kaggle/input/openmind-brain-mri/OpenMind_processed/MRA/
   /kaggle/input/deeplesion-brain-ct/NIH_deeplesion/CT/
   ```

---

### Phase 3: Modify Notebook for Kaggle Environment

#### Step 3.1: Update Data Paths

**Current notebook** (`notebooks/phase0_pretrain_kaggle.ipynb` cell 7):
```python
# ❌ OLD (template paths)
config.mri_dirs = [
    "/kaggle/input/openmind-mri/MRI_T1",
    "/kaggle/input/openmind-mri/MRI_T2",
    "/kaggle/input/openmind-mri/MRA",
]
config.ct_dirs = [
    "/kaggle/input/deeplesion-ct/CT",
]
```

**NEW (actual Kaggle dataset paths)**:
```python
# ✅ UPDATED (match actual uploaded dataset structure)
config.mri_dirs = [
    "/kaggle/input/openmind-brain-mri/OpenMind_processed/MRI_T1",
    "/kaggle/input/openmind-brain-mri/OpenMind_processed/MRI_T2",
    "/kaggle/input/openmind-brain-mri/OpenMind_processed/MRA",
]
config.ct_dirs = [
    "/kaggle/input/deeplesion-brain-ct/NIH_deeplesion/CT",
]
```

#### Step 3.2: Fix Code Import Method

**Current notebook** (cell 5) tries to copy from uploaded dataset:
```python
# ❌ PROBLEM: Requires uploading source as separate Kaggle dataset
CODE_PATH = "/kaggle/input/rsna-phase0-source/source"
```

**BETTER: Clone directly from GitHub**:
```python
# ✅ SOLUTION: Clone from GitHub (always up-to-date)
import os
import sys

# Clone repository if not exists
if not os.path.exists("/kaggle/working/RSNA-2025"):
    print("Cloning repository from GitHub...")
    !git clone https://github.com/YOUR_USERNAME/RSNA-2025.git /kaggle/working/RSNA-2025
    print("✅ Repository cloned")
else:
    print("✅ Repository already exists")

# Add to Python path
sys.path.insert(0, "/kaggle/working/RSNA-2025")

# Verify imports
try:
    from source.modules.phase0.models.pretrainer import WaveFormerSparKMiMPretrainer
    from source.modules.phase0.data.unified_dataloaders import create_unified_dataloaders
    from source.modules.phase0.utils.checkpoint import CheckpointManager
    from source.config.phase0_config import get_config
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    raise
```

#### Step 3.3: Install Required Packages (INCLUDING MinkowskiEngine)

**CRITICAL: MinkowskiEngine is REQUIRED** - Pure SparK sparse operations, NO fallback!

```python
# Step 1: Install standard dependencies
!pip install -q ptwt==0.1.9 nibabel==5.2.1 tqdm==4.66.2

# Step 2: Build MinkowskiEngine from source (cuda12-installation branch)
!git clone https://github.com/CiSong10/MinkowskiEngine.git /tmp/MinkowskiEngine
!cd /tmp/MinkowskiEngine && git checkout cuda12-installation && python setup.py install --force_cuda

# Verify installations
import ptwt
import nibabel as nib
import tqdm
import MinkowskiEngine as ME

print(f"✅ ptwt: {ptwt.__version__}")
print(f"✅ nibabel: {nib.__version__}")
print(f"✅ tqdm: {tqdm.__version__}")
print(f"✅ MinkowskiEngine: {ME.__version__}")
print(f"✅ CUDA support: {ME.is_cuda_available()}")
```

**Expected build time**: 5-10 minutes
**Requirements**: Kaggle GPU enabled (for CUDA compilation)

---

### Phase 4: Kaggle-Specific Configuration

#### Step 4.1: GPU Settings

Kaggle provides:
- **Free tier**: T4 GPU (16GB VRAM), 30 hours/week
- **Expert tier**: P100 GPU (16GB VRAM), 30 hours/week

**Optimal Configuration for Kaggle T4**:
```python
# In notebook cell 7, after config = get_config('kaggle'):

# Kaggle T4 optimized settings
config.img_size = (64, 64, 64)      # Smaller than production (128³)
config.patch_size = 8                # Keep same
config.embed_dim = 512               # Reduce from 768
config.depth = 8                     # Reduce from 12
config.num_heads = 8                 # Keep same
config.batch_size_mri = 4            # Conservative for 16GB
config.batch_size_ct = 2             # CT uses 3x memory
config.num_workers = 2               # Kaggle has limited CPU
config.learning_rate = 1e-4

print("Kaggle T4 Configuration:")
print(f"  Image size: {config.img_size}")
print(f"  Model params: ~50M (estimated)")
print(f"  GPU memory: ~8-10GB expected")
print(f"  Batch sizes: MRI={config.batch_size_mri}, CT={config.batch_size_ct}")
```

#### Step 4.2: Training Duration

**Kaggle Time Limits**:
- Maximum session: 9 hours (then auto-shutdown)
- Weekly quota: 30 GPU hours

**Strategy**:
```python
# Adjust epochs based on available time
NUM_EPOCHS = 30  # Conservative for 9-hour limit

# Estimate:
# - Steps per epoch: ~(5573+671)/batch_size = ~1000 steps
# - Time per step: ~2-3 seconds
# - Time per epoch: ~30-45 minutes
# - 30 epochs: ~15-22 hours (need 3 sessions)

# PLAN: Run multiple sessions with checkpointing
# Session 1: Epochs 1-10
# Session 2: Epochs 11-20
# Session 3: Epochs 21-30
```

#### Step 4.3: Checkpoint Resuming

**Add to notebook** (new cell before training loop):
```python
# Check for existing checkpoint and resume
import glob

checkpoint_files = glob.glob("/kaggle/working/checkpoints/checkpoint_epoch_*.pth")
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Found checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))

    print(f"✅ Resuming from epoch {start_epoch}")
else:
    start_epoch = 0
    best_loss = float('inf')
    print("✅ Starting fresh training")

# Modify training loop
for epoch in range(start_epoch, NUM_EPOCHS):  # ← Changed from range(NUM_EPOCHS)
    # ... rest of training loop
```

---

### Phase 5: Execution Plan

#### Session 1: Initial Setup and Validation (2-3 hours)

**Steps**:
1. Create Kaggle notebook
2. Clone GitHub repository
3. Install packages
4. Load small subset (50 MRI + 50 CT)
5. Run 5 epochs to validate setup
6. Verify loss decreasing

**Expected Output**:
```
Epoch 1: Loss ~2000
Epoch 2: Loss ~1000
Epoch 3: Loss ~600
Epoch 4: Loss ~500
Epoch 5: Loss ~400
```

**Validation Checklist**:
- [ ] No import errors
- [ ] Both MRI and CT batches loading
- [ ] Loss decreasing
- [ ] Checkpoints saving
- [ ] GPU utilization >80%

#### Session 2: Full Training Part 1 (9 hours)

**Modify notebook**:
```python
# Remove subset limits
# MAX_MRI_SAMPLES = 50  ← Delete or comment out
# MAX_CT_SAMPLES = 50   ← Delete or comment out

NUM_EPOCHS = 10  # First batch of full training
```

**Expected**:
- Time: ~6-8 hours
- Loss: ~2000 → ~200
- Checkpoints: epochs 1-10

**At end of session**:
1. Download checkpoints folder
2. Upload to Kaggle dataset for next session

#### Session 3: Full Training Part 2 (9 hours)

**Setup**:
1. Upload previous checkpoints as Kaggle dataset
2. Add dataset to notebook
3. Resume from epoch 10

```python
NUM_EPOCHS = 20  # Continue to epoch 20
```

#### Session 4+: Continue Until Convergence

**Repeat until**:
- Loss < 100
- Or loss plateaus for 5+ epochs

---

## 🔧 ENVIRONMENT COMPARISON

### Docker (Local) vs Kaggle (Cloud)

| Component | Docker (Local) | Kaggle (Cloud) | Match? |
|-----------|---------------|----------------|--------|
| PyTorch | 2.4.0 | 2.4.0 (pre-installed) | ✅ Yes |
| CUDA | 12.1 | 12.1 (pre-installed) | ✅ Yes |
| Python | 3.10 | 3.10 | ✅ Yes |
| ptwt | 0.1.9 | Install via pip | ✅ Yes |
| nibabel | 5.2.1 | Install via pip | ✅ Yes |
| MinkowskiEngine | Installed (cuda12) | **REQUIRED** (build from source) | ✅ Yes |
| GPU | RTX 5000 (16GB) | T4 (16GB) | ✅ Similar |

### Key Differences

**Docker Advantages**:
- Full system control
- Can install MinkowskiEngine via Docker build
- Persistent storage
- No time limits

**Kaggle Advantages**:
- Free GPU access (30 hours/week)
- Pre-configured environment
- No Docker complexity
- Easy sharing/collaboration
- Auto-save to cloud

**Compatibility**: ✅ 100% compatible
- Both use PyTorch 2.4.0 + CUDA 12.1
- Same Python packages (ptwt, nibabel)
- Same model code (pure SparK with MinkowskiEngine)
- Same data format (NIfTI)

---

## 📝 UPDATED KAGGLE NOTEBOOK


### Corrected Cell 2: Install Packages (WITH MinkowskiEngine)

```python
# Step 1: Install standard dependencies
!pip install -q ptwt==0.1.9 nibabel==5.2.1 tqdm==4.66.2

# Step 2: Build MinkowskiEngine from source
!git clone https://github.com/CiSong10/MinkowskiEngine.git /tmp/MinkowskiEngine
!cd /tmp/MinkowskiEngine && git checkout cuda12-installation && python setup.py install --force_cuda

# Verify
import ptwt, nibabel, tqdm, MinkowskiEngine as ME
print(f"✅ ptwt {ptwt.__version__}")
print(f"✅ nibabel {nibabel.__version__}")
print(f"✅ tqdm {tqdm.__version__}")
print(f"✅ MinkowskiEngine {ME.__version__}")
print(f"✅ CUDA support: {ME.is_cuda_available()}")
```

**Build time**: ~5-10 minutes

### Corrected Cell 5: Clone from GitHub

```python
import os
import sys

# Clone repository (replace with your GitHub URL)
REPO_URL = "https://github.com/YOUR_USERNAME/RSNA-2025.git"

if not os.path.exists("/kaggle/working/RSNA-2025"):
    print("📥 Cloning repository from GitHub...")
    !git clone {REPO_URL} /kaggle/working/RSNA-2025
    print("✅ Repository cloned")
else:
    print("✅ Repository already cloned")

# Add to Python path
sys.path.insert(0, "/kaggle/working/RSNA-2025")

# Verify imports
try:
    from source.modules.phase0.models.pretrainer import WaveFormerSparKMiMPretrainer
    from source.modules.phase0.data.unified_dataloaders import create_unified_dataloaders
    from source.modules.phase0.utils.checkpoint import CheckpointManager
    from source.config.phase0_config import get_config
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Check that source/ folder has correct structure")
    raise
```

### Corrected Cell 7: Configuration

```python
# Get Kaggle-optimized config
config = get_config('dev')  # Use dev config as base

# Override with actual Kaggle dataset paths
# TODO: Replace with your actual Kaggle dataset names
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

print("✅ Kaggle Configuration:")
print(f"   Image size: {config.img_size}")
print(f"   Embed dim: {config.embed_dim}")
print(f"   Depth: {config.depth}")
print(f"   Batch sizes: MRI={config.batch_size_mri}, CT={config.batch_size_ct}")
```

### New Cell: Checkpoint Resuming

```python
# Check for existing checkpoints and resume if available
import glob

checkpoint_dir = "/kaggle/working/checkpoints"
checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pth")

if checkpoint_files:
    # Resume from latest checkpoint
    latest_checkpoint = max(checkpoint_files,
                           key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"📂 Found checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('loss', float('inf'))

    print(f"✅ Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
else:
    start_epoch = 0
    best_loss = float('inf')
    print("✅ Starting fresh training (no checkpoint found)")
```

---

## ⚡ QUICK START COMMANDS

### 1. Prepare Local Repository

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
EOF

# Commit and push
git add .
git commit -m "Phase 0: Kaggle-ready unified multi-modal implementation"
git push origin main
```

### 2. Upload Data to Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Upload MRI dataset (from local)
cd /home/thanhjash/RSNA/data/processed/openmind
kaggle datasets create -p . -t "OpenMind Brain MRI"

# Upload CT dataset
cd /home/thanhjash/RSNA/data/processed/NIH_deeplesion
kaggle datasets create -p . -t "DeepLesion Brain CT"
```

### 3. Create Kaggle Notebook

1. Go to https://kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → GPU T4
4. Add data sources (your uploaded datasets)
5. First cell:
```python
!git clone https://github.com/YOUR_USERNAME/RSNA-2025.git /kaggle/working/RSNA-2025
```

### 4. Run Training

- Click "Run All"
- Monitor progress
- Download checkpoints before session expires

---

## 🎯 SUCCESS METRICS

### Validation (First Session)
- [ ] Training starts without errors
- [ ] Loss decreases (2000 → 400 in 5 epochs)
- [ ] Both MRI and CT batches processing
- [ ] GPU utilization >80%
- [ ] Checkpoints saving successfully

### Full Training (Multiple Sessions)
- [ ] Final loss < 100
- [ ] Total training time < 30 hours (within Kaggle quota)
- [ ] Model checkpoints saved and downloadable
- [ ] Loss curve shows steady decrease

---

## 📚 REFERENCE FILES

1. **GROUND_TRUTH.md** - Implementation details
2. **DOCKER_GUIDE.md** - Local Docker environment (for reference)
3. **PHASE0_MODEL_ARCHITECTURE.md** - Model architecture
4. **IMPLEMENTATION_STATUS_REPORT.md** - Current status
5. **notebooks/phase0_pretrain_kaggle.ipynb** - Kaggle notebook (update cell 5 and 7)

---

## 🆘 TROUBLESHOOTING

### Issue: Import Error in Kaggle

**Symptom**: `ModuleNotFoundError: No module named 'source'`

**Solution**:
```python
# Verify repository cloned
!ls -la /kaggle/working/RSNA-2025/

# Check Python path
import sys
print(sys.path)

# Ensure path added
sys.path.insert(0, "/kaggle/working/RSNA-2025")
```

### Issue: Data Not Found

**Symptom**: `FileNotFoundError` or empty datasets

**Solution**:
```python
# Verify dataset paths
!ls -la /kaggle/input/

# Check actual structure
!ls -la /kaggle/input/openmind-brain-mri/
!ls -la /kaggle/input/deeplesion-brain-ct/

# Update config.mri_dirs and config.ct_dirs accordingly
```

### Issue: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size
config.batch_size_mri = 2  # From 4
config.batch_size_ct = 1   # From 2

# Or reduce model size
config.embed_dim = 256     # From 512
config.depth = 4           # From 8
```

---

**FINAL NOTE**: Kaggle environment is **fully compatible** with local Docker environment. No Docker needed on Kaggle!
