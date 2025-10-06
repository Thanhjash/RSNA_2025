# Docker Guide - Phase 0 Pre-training Environment

**Last Updated**: 2025-10-06
**Image**: `rsna-minkowski:final`
**Status**: ✅ Validated and Working

---

## Quick Start

### Subset Validation (TESTED ✅)
```bash
docker run --rm --gpus all \
  -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final \
  python /workspace/rsna/source/train_phase0_subset.py 2>&1 | \
  tee subset_validation.log
```

**Expected**: 10 epochs, ~10-15 minutes, loss decreases from ~2000 → ~300

### Full Production Training (READY)
```bash
# Edit train_phase0_subset.py first:
# - Remove MAX_MRI_SAMPLES and MAX_CT_SAMPLES limits
# - Set NUM_EPOCHS = 100
# - Optionally use production config

docker run --rm --gpus all \
  -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final \
  python /workspace/rsna/source/train_phase0_subset.py 2>&1 | \
  tee production_training.log
```

**Expected**: 100 epochs, 24-48 hours, final loss < 100

---

## Docker Environment Details

### Image Specifications

**Name**: `rsna-minkowski:final`
**Base**: NVIDIA PyTorch 24.02-py3
**Dockerfile**: `source/MinkowskiEngine/docker/Dockerfile.final`

**Key Components**:
- PyTorch 2.4.0
- CUDA 12.1
- MinkowskiEngine (installed, using dense fallback)
- ptwt (PyTorch Wavelet Toolbox)
- nibabel (NIfTI file handling)

**Build Command** (if needed):
```bash
cd /home/thanhjash/RSNA/source/MinkowskiEngine/docker
docker build -f Dockerfile.final -t rsna-minkowski:final .
```

---

## Volume Mounting

### Required Mount
```bash
-v /home/thanhjash/RSNA:/workspace/rsna
```

**Host Path**: `/home/thanhjash/RSNA` (your project root)
**Container Path**: `/workspace/rsna` (working directory in container)

### Directory Structure in Container
```
/workspace/rsna/
├── data/processed/
│   ├── openmind/OpenMind_processed/
│   │   ├── MRI_T1/  # 4,288 volumes
│   │   ├── MRI_T2/  # 1,222 volumes
│   │   └── MRA/     # 63 volumes
│   └── NIH_deeplesion/
│       └── CT/      # 671 volumes
├── source/
│   ├── train_phase0_subset.py  # Main entry point
│   ├── modules/phase0/         # Core implementation
│   └── config/                 # Configuration
├── checkpoints/                # Auto-created during training
└── *.log                       # Training logs
```

---

## Common Docker Commands

### Interactive Shell (Debugging)
```bash
docker run --rm -it --gpus all \
  -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final \
  bash
```

Inside container:
```bash
# Check GPU
nvidia-smi

# Test Python environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run training manually
cd /workspace/rsna
python source/train_phase0_subset.py
```

### Background Training (Detached)
```bash
docker run -d --name phase0_training --gpus all \
  -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final \
  python /workspace/rsna/source/train_phase0_subset.py
```

Monitor:
```bash
# View logs
docker logs -f phase0_training

# Check if running
docker ps

# Stop training
docker stop phase0_training

# Remove container
docker rm phase0_training
```

### Check GPU Usage
```bash
# From host
nvidia-smi

# Inside running container
docker exec phase0_training nvidia-smi
```

---

## Training Configurations

### Subset Validation (Default in train_phase0_subset.py)
```python
MAX_MRI_SAMPLES = 50
MAX_CT_SAMPLES = 50
NUM_EPOCHS = 10
config = get_config('dev')

# Image size: 32³
# Batch sizes: MRI=2, CT=2
# Model: 7.6M parameters
# GPU memory: ~2GB
# Time: 10-15 minutes
```

### Full Training (Modify train_phase0_subset.py)
```python
# Remove or comment out:
# MAX_MRI_SAMPLES = 50
# MAX_CT_SAMPLES = 50

NUM_EPOCHS = 100
config = get_config('production')

# Image size: 128³
# Batch sizes: MRI=8, CT=4
# Model: ~136M parameters
# GPU memory: 8-12GB
# Time: 24-48 hours
```

### Memory-Constrained Training
```python
# Use subset config for lower memory
config = get_config('dev')
config.batch_size_mri = 2
config.batch_size_ct = 1
config.embed_dim = 256  # Instead of 768
config.depth = 8        # Instead of 12
```

---

## Monitoring Training

### Real-time Monitoring
```bash
# Follow log file
tail -f subset_validation.log

# Watch GPU usage
watch -n 1 nvidia-smi

# Both in split terminal
tmux
# Pane 1: tail -f subset_validation.log
# Pane 2: watch -n 1 nvidia-smi
```

### Check Training Progress
```bash
# Count completed epochs
grep "Epoch.*Summary" subset_validation.log | wc -l

# View latest epoch summary
grep "Epoch.*Summary" -A 5 subset_validation.log | tail -10

# Extract loss values
grep "Average Total Loss" subset_validation.log
```

### Check Checkpoints
```bash
ls -lh checkpoints/phase0_subset_validation/
# Should see:
# - checkpoint_epoch_*.pth
# - best_model.pth
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or model size
```python
# In train_phase0_subset.py, before model creation:
config.batch_size_mri = 2  # Reduce from 8
config.batch_size_ct = 1   # Reduce from 4
config.embed_dim = 256     # Reduce from 768
```

### Issue: Data Not Found
**Symptom**: "Directory not found" warnings

**Solution**: Check mount paths
```bash
# From host:
ls /home/thanhjash/RSNA/data/processed/openmind/
ls /home/thanhjash/RSNA/data/processed/NIH_deeplesion/

# In container:
docker run --rm -v /home/thanhjash/RSNA:/workspace/rsna rsna-minkowski:final \
  ls /workspace/rsna/data/processed/openmind/
```

### Issue: Import Errors
**Symptom**: `ModuleNotFoundError`

**Solution**: Check Python path
```python
# In train_phase0_subset.py, verify:
sys.path.insert(0, str(Path(__file__).parent))
```

### Issue: Container Won't Start
**Check**:
```bash
# GPU available?
nvidia-smi

# Docker has GPU support?
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Image exists?
docker images | grep rsna-minkowski
```

---

## Performance Tuning

### Maximize GPU Utilization
```python
# Increase batch size until CUDA OOM
config.batch_size_mri = 16  # Try increasing
config.batch_size_ct = 8    # CT uses 3x memory

# Use larger model
config.embed_dim = 768
config.depth = 12

# Reduce number of workers if CPU bottleneck
config.num_workers = 2  # Or 4
```

### Reduce Training Time
```python
# Mixed precision (FP16) - if implemented
# Enable in model init or training loop

# Gradient accumulation
accumulation_steps = 4
# Modify training loop to accumulate gradients

# Reduce data loading overhead
config.num_workers = 4  # Increase if CPU allows
```

### Checkpoint Management
```bash
# Keep only best checkpoint (save disk space)
# In checkpointing code, set:
max_keep_checkpoints = 1

# Compress checkpoints
tar -czf checkpoints_backup.tar.gz checkpoints/

# Clean old runs
rm -rf checkpoints/old_run_*
```

---

## Data Pipeline in Docker

### Data Locations (Container Paths)
```
MRI Data:
/workspace/rsna/data/processed/openmind/OpenMind_processed/MRI_T1/*.nii.gz
/workspace/rsna/data/processed/openmind/OpenMind_processed/MRI_T2/*.nii.gz
/workspace/rsna/data/processed/openmind/OpenMind_processed/MRA/*.nii.gz

CT Data:
/workspace/rsna/data/processed/NIH_deeplesion/CT/*.nii.gz
```

### Data Format
- **MRI**: 1-channel NIfTI, 1mm³ isotropic, RAS orientation
- **CT**: 3-channel NIfTI (brain/blood/bone windows), 1mm³ isotropic, RAS orientation

### Data Shape Handling
- **MRI**: `(D, H, W)` → automatically adds channel dim → `(1, D, H, W)`
- **CT**: `(D, H, W, 1, 3)` → squeeze + transpose → `(3, D, H, W)`

---

## Outputs and Results

### Training Outputs
```
checkpoints/phase0_subset_validation/
├── checkpoint_epoch_1.pth
├── checkpoint_epoch_2.pth
├── ...
├── checkpoint_epoch_10.pth
└── best_model.pth            # Lowest loss checkpoint

subset_validation.log         # Complete training log
```

### Loading Trained Model
```python
import torch
from source.modules.phase0.models.pretrainer import WaveFormerSparKMiMPretrainer

# Load checkpoint
checkpoint = torch.load('checkpoints/phase0_subset_validation/best_model.pth')

# Recreate model
model = WaveFormerSparKMiMPretrainer(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Extract encoder only
encoder = model.waveformer
torch.save(encoder.state_dict(), 'waveformer_encoder.pth')
```

---

## Production Deployment Checklist

### Before Full Training
- [ ] Subset validation completed successfully
- [ ] Loss decreased steadily (82% reduction verified)
- [ ] Both MRI and CT batches processed without errors
- [ ] Checkpoints saving correctly
- [ ] GPU memory usage acceptable (<90%)
- [ ] Disk space available (>100GB for checkpoints)

### Modify for Production
```python
# In train_phase0_subset.py:

# 1. Remove subset limits
# MAX_MRI_SAMPLES = 50  ← Comment out or delete
# MAX_CT_SAMPLES = 50   ← Comment out or delete

# 2. Set production epochs
NUM_EPOCHS = 100  # Or 200 if time permits

# 3. Use production config (optional)
config = get_config('production')  # Instead of 'dev'

# 4. Update checkpoint directory
checkpoint_dir = Path("checkpoints/phase0_production")
```

### Launch Production
```bash
# Screen or tmux session (survives disconnection)
screen -S phase0_training

# Or tmux:
tmux new -s phase0_training

# Run training
docker run --rm --gpus all \
  -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final \
  python /workspace/rsna/source/train_phase0_subset.py 2>&1 | \
  tee production_training.log

# Detach: Ctrl+A, D (screen) or Ctrl+B, D (tmux)
# Reattach: screen -r phase0_training or tmux attach -t phase0_training
```

### Monitor Production
```bash
# Check progress every few hours
tail -100 production_training.log | grep "Epoch.*Summary" -A 5

# GPU usage
nvidia-smi

# Disk space
df -h

# Estimated time remaining
# (Total epochs - Current epoch) × Time per epoch
```

---

## Quick Reference Commands

### Start Subset Training
```bash
docker run --rm --gpus all -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final python /workspace/rsna/source/train_phase0_subset.py \
  2>&1 | tee subset_validation.log
```

### Start Production Training
```bash
# After modifying train_phase0_subset.py
screen -S phase0
docker run --rm --gpus all -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final python /workspace/rsna/source/train_phase0_subset.py \
  2>&1 | tee production_training.log
# Ctrl+A, D to detach
```

### Monitor Training
```bash
tail -f production_training.log
watch -n 1 nvidia-smi
grep "Average Total Loss" production_training.log | tail -10
```

### Extract Results
```bash
# Best loss achieved
grep "⭐ Best model saved" production_training.log | tail -1

# All epoch summaries
grep "Epoch.*Summary" -A 6 production_training.log > training_summary.txt

# Loss progression
grep "Average Total Loss:" production_training.log | \
  awk '{print $4}' > loss_values.txt
```

---

## Environment Variables (Optional)

```bash
# Set GPU device
docker run --rm --gpus '"device=0"' \
  -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final \
  python /workspace/rsna/source/train_phase0_subset.py

# Set PyTorch settings
docker run --rm --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  -v /home/thanhjash/RSNA:/workspace/rsna \
  rsna-minkowski:final \
  python /workspace/rsna/source/train_phase0_subset.py
```

---

## FAQs

**Q: How much disk space needed?**
A: ~150GB total
- Data: ~50GB (MRI + CT)
- Checkpoints: ~50-100GB (depends on model size and frequency)
- Docker image: ~10GB

**Q: Can I pause and resume training?**
A: Yes, modify training script to:
1. Save checkpoint every epoch
2. Load checkpoint if exists at start
3. Resume from saved epoch number

**Q: How to use multiple GPUs?**
A: Modify model initialization:
```python
model = nn.DataParallel(model, device_ids=[0, 1])
# Or use DistributedDataParallel for better performance
```

**Q: Training slower than expected?**
A: Check:
- GPU utilization (`nvidia-smi` should show ~90-100%)
- Data loading (increase `num_workers`)
- Batch size (increase if GPU memory allows)
- Disk I/O (use SSD for data)

**Q: How to know if training is working correctly?**
A: Checklist:
- ✅ Loss decreasing (at least overall trend)
- ✅ Both MRI and CT batches processing
- ✅ Contrastive loss > 0 (not zero)
- ✅ No CUDA errors or NaN losses
- ✅ Checkpoints saving successfully

---

**For complete implementation details, see**: `GROUND_TRUTH.md`
**For architecture details, see**: `PHASE0_MODEL_ARCHITECTURE.md`
**For training results, see**: `IMPLEMENTATION_STATUS_REPORT.md`
