#!/usr/bin/env python3
"""
SAFE DeepLesion Preprocessing Pipeline for RSNA 2025
Memory-optimized version to prevent system crashes

This script processes the DeepLesion dataset with conservative memory usage
and harmonizes it with RSNA and OpenMind pipeline outputs.

Key Safety Features:
- Conservative multiprocessing (max 4 workers)
- Batch processing with memory cleanup
- Resume capability for interrupted processing  
- Resource monitoring and early termination
- Extensive error handling and validation

Standardization Features:
- RAS orientation (1.0x1.0x1.0mm spacing)
- 3-channel CT windowing (brain/blood/bone)
- NIfTI output matching RSNA format
"""

import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import time
import logging
import warnings
import imageio.v2 as imageio
from collections import defaultdict
import gc
import psutil
import os
import argparse
import subprocess
import signal

# Try to import nvidia-ml-py for GPU monitoring (optional)
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sitk.ProcessObject.SetGlobalWarningDisplay(False)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Input paths
DEEPLESION_RAW_DIR = Path("data/raw/NIH_deeplesion")
PNG_DIR = DEEPLESION_RAW_DIR / "minideeplesion"
METADATA_CSV = DEEPLESION_RAW_DIR / "DL_info.csv"

# Output path
OUTPUT_DIR = Path("data/processed/NIH_deeplesion")

# Harmonization settings (MUST MATCH other pipelines)
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_ORIENTATION = "RAS"
CT_WINDOWS = {
    "brain": {"level": 40, "width": 80},
    "blood": {"level": 80, "width": 200},
    "bone": {"level": 600, "width": 3000},
}

# Safety settings (GPU-friendly)
MAX_WORKERS = 2  # Very conservative to not interfere with GPU training
BATCH_SIZE = 25  # Smaller batches to minimize memory impact
MEMORY_THRESHOLD = 70  # Stop if memory usage > 70%
GPU_THRESHOLD = 50  # Stop if GPU memory > 50% (if CUDA available)
MIN_SLICES_PER_SERIES = 3  # Minimum slices for valid 3D volume
CPU_PRIORITY = "low"  # Set low CPU priority

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir):
    """Setup logging with detailed formatting."""
    log_path = output_dir / "deeplesion_safe_processing.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# =============================================================================
# RESOURCE MONITORING
# =============================================================================
def check_memory_usage():
    """Check current memory usage percentage."""
    return psutil.virtual_memory().percent

def check_gpu_usage():
    """Check GPU memory usage if available."""
    if not NVML_AVAILABLE:
        return 0, "GPU monitoring not available"
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_percent = (mem_info.used / mem_info.total) * 100
        return gpu_percent, f"GPU memory: {gpu_percent:.1f}%"
    except:
        return 0, "GPU monitoring failed"

def check_system_health():
    """Check if system is healthy enough to continue processing."""
    memory_percent = check_memory_usage()
    gpu_percent, gpu_status = check_gpu_usage()
    
    if memory_percent > MEMORY_THRESHOLD:
        return False, f"Memory usage too high: {memory_percent:.1f}%"
    
    if gpu_percent > GPU_THRESHOLD and NVML_AVAILABLE:
        return False, f"GPU usage too high: {gpu_percent:.1f}% (training model detected)"
    
    return True, f"System OK - RAM: {memory_percent:.1f}%, {gpu_status}"

def set_low_priority():
    """Set low CPU priority to not interfere with GPU training."""
    try:
        # Set low priority on Linux/Unix
        os.nice(10)
        # Set process to use only non-GPU cores if possible
        if hasattr(os, 'sched_setaffinity'):
            # Use only some CPU cores, leaving others for GPU training
            available_cpus = list(range(min(4, os.cpu_count())))
            os.sched_setaffinity(0, available_cpus)
    except:
        pass

# =============================================================================
# DATA VALIDATION
# =============================================================================
def validate_series_files(series_uid, slice_paths, logger):
    """Validate that series has sufficient slices and readable files."""
    if len(slice_paths) < MIN_SLICES_PER_SERIES:
        logger.debug(f"Series {series_uid}: Insufficient slices ({len(slice_paths)} < {MIN_SLICES_PER_SERIES})")
        return False, f"Only {len(slice_paths)} slices (minimum {MIN_SLICES_PER_SERIES} required)"
    
    # Check if files are readable
    readable_count = 0
    for path in slice_paths:
        try:
            if path.exists() and path.stat().st_size > 0:
                readable_count += 1
        except:
            continue
    
    if readable_count < MIN_SLICES_PER_SERIES:
        return False, f"Only {readable_count} readable slices"
    
    return True, f"Valid series with {len(slice_paths)} slices"

# =============================================================================
# WORKER FUNCTION (MEMORY-OPTIMIZED)
# =============================================================================
def process_series_worker(args):
    """
    Memory-optimized worker function for processing one 3D series.
    """
    series_uid, slice_paths, spacing_info, output_dir_str = args
    output_dir = Path(output_dir_str)
    
    result = {
        "series_uid": series_uid,
        "status": "FAILED",
        "message": "",
        "processing_time": 0,
        "num_slices": len(slice_paths),
        "memory_peak": 0
    }
    
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    try:
        # Check if already processed
        final_path = output_dir / "CT" / f"{series_uid}.nii.gz"
        if final_path.exists():
            result["status"] = "SKIPPED"
            result["message"] = "Already processed"
            result["processing_time"] = time.time() - start_time
            return result

        # Step 1: Load and sort slices with memory awareness
        slices = []
        slice_indices = []
        
        # Sort paths by slice index first to avoid memory fragmentation
        slice_data = []
        for slice_path in slice_paths:
            try:
                slice_idx = int(slice_path.stem)  # Get slice number from filename
                slice_data.append((slice_idx, slice_path))
            except (ValueError, IndexError):
                continue
        
        if len(slice_data) < MIN_SLICES_PER_SERIES:
            result["message"] = f"Insufficient valid slices after parsing: {len(slice_data)}"
            return result
            
        # Sort by slice index
        slice_data.sort(key=lambda x: x[0])
        
        # Load slices in order
        for slice_idx, slice_path in slice_data:
            try:
                slice_img = imageio.imread(slice_path)
                if slice_img is not None:
                    slices.append(slice_img)
                    slice_indices.append(slice_idx)
            except Exception as e:
                continue  # Skip corrupted slices
        
        if len(slices) < MIN_SLICES_PER_SERIES:
            result["message"] = f"Too few readable slices: {len(slices)}"
            return result

        # Step 2: Stack into 3D volume
        try:
            volume_np = np.stack(slices, axis=0)
            # Clear slice list to free memory
            del slices
            gc.collect()
        except Exception as e:
            result["message"] = f"Failed to stack slices: {e}"
            return result

        # Step 3: Convert to Hounsfield Units (critical for DeepLesion)
        volume_np = volume_np.astype(np.float32) - 32768.0
        
        # Monitor memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        result["memory_peak"] = max(result["memory_peak"], current_memory - initial_memory)
        
        # Step 4: Create SimpleITK image with spacing
        img_sitk = sitk.GetImageFromArray(volume_np)
        
        try:
            # Parse spacing: "x, y, z" format from CSV
            spacing_parts = [float(s.strip()) for s in spacing_info.split(',')]
            if len(spacing_parts) >= 3:
                # SimpleITK expects [x, y, z] order
                initial_spacing = tuple(spacing_parts[:3])
                img_sitk.SetSpacing(initial_spacing)
            else:
                raise ValueError(f"Invalid spacing format: {spacing_info}")
        except Exception as e:
            result["message"] = f"Invalid spacing '{spacing_info}': {e}"
            return result

        # Step 5: Reorient to RAS (with fallback)
        try:
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(TARGET_ORIENTATION)
            oriented_img = orient_filter.Execute(img_sitk)
        except:
            oriented_img = img_sitk  # Use original if orientation fails
        
        # Clear original image
        del img_sitk
        del volume_np
        gc.collect()

        # Step 6: Resample to target spacing
        try:
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(TARGET_SPACING)
            resampler.SetInterpolator(sitk.sitkLinear)
            
            original_size = oriented_img.GetSize()
            original_spacing = oriented_img.GetSpacing()
            
            # Calculate new size with safety bounds
            new_size = []
            for osz, ospc, nspc in zip(original_size, original_spacing, TARGET_SPACING):
                new_dim = max(1, int(round(osz * ospc / nspc)))
                # Safety limit to prevent excessive memory usage
                new_dim = min(new_dim, 1024)  
                new_size.append(new_dim)
            
            resampler.SetSize(new_size)
            resampler.SetOutputOrigin(oriented_img.GetOrigin())
            resampler.SetOutputDirection(oriented_img.GetDirection())
            resampled_img_sitk = resampler.Execute(oriented_img)
            
        except Exception as e:
            result["message"] = f"Resampling failed: {e}"
            return result
        
        # Clear oriented image
        del oriented_img
        gc.collect()

        # Step 7: Apply multi-channel CT windowing (same as RSNA)
        img_np = sitk.GetArrayFromImage(resampled_img_sitk)
        
        if np.isnan(img_np).any() or np.isinf(img_np).any():
            result["message"] = "Invalid pixel values after resampling"
            return result

        # Create 3 channels
        channels = []
        for w in CT_WINDOWS.values():
            clipped = np.clip(img_np, w["level"] - w["width"] // 2, w["level"] + w["width"] // 2)
            c_min, c_max = clipped.min(), clipped.max()
            if c_max > c_min:
                normalized = (clipped - c_min) / (c_max - c_min)
            else:
                normalized = np.zeros_like(clipped)
            channels.append(normalized.astype(np.float32))
        
        # Stack channels and create final image
        final_np = np.stack(channels, axis=-1)
        final_img = sitk.GetImageFromArray(final_np, isVector=True)
        final_img.CopyInformation(resampled_img_sitk)
        
        # Clear intermediate arrays
        del img_np, channels, final_np, resampled_img_sitk
        gc.collect()

        # Step 8: Save final image
        modality_dir = output_dir / "CT"
        modality_dir.mkdir(parents=True, exist_ok=True)
        
        sitk.WriteImage(final_img, str(final_path))
        
        # Final cleanup
        del final_img
        gc.collect()
        
        result["status"] = "SUCCESS"
        result["message"] = f"Processed {len(slice_indices)} slices"
        
    except Exception as e:
        result["message"] = f"{type(e).__name__}: {str(e)}"
    
    finally:
        result["processing_time"] = time.time() - start_time
        # Final memory measurement
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        result["memory_peak"] = max(result["memory_peak"], final_memory - initial_memory)
    
    return result

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================
def process_batch(tasks, batch_num, total_batches, logger):
    """Process a batch of tasks with memory monitoring."""
    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(tasks)} series)")
    
    # Check system health before starting batch
    healthy, status = check_system_health()
    logger.info(f"System status: {status}")
    
    if not healthy:
        logger.warning("System health check failed, skipping batch")
        return []
    
    # Conservative worker count
    num_workers = min(MAX_WORKERS, len(tasks), mp.cpu_count() // 2)
    logger.info(f"Using {num_workers} workers for batch processing")
    
    batch_results = []
    try:
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=len(tasks), desc=f"Batch {batch_num}") as pbar:
                for result in pool.imap_unordered(process_series_worker, tasks):
                    batch_results.append(result)
                    pbar.update(1)
                    
                    # Periodic health check during batch
                    if len(batch_results) % 10 == 0:
                        healthy, status = check_system_health()
                        if not healthy:
                            logger.warning(f"Health check failed during batch: {status}")
                            pool.terminate()
                            break
                        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
    
    # Force cleanup after batch
    gc.collect()
    
    return batch_results

def get_already_processed(output_dir):
    """Get set of already processed series UIDs."""
    processed = set()
    ct_dir = output_dir / "CT"
    if ct_dir.exists():
        for nifti_file in ct_dir.glob("*.nii.gz"):
            processed.add(nifti_file.stem)
    return processed

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main orchestration with batch processing and resume capability."""
    
    parser = argparse.ArgumentParser(description="Safe DeepLesion Preprocessing")
    parser.add_argument("--test", action="store_true", help="Test mode: process only first 10 series")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--gpu-safe", action="store_true", default=True, help="GPU-safe mode with low priority")
    args = parser.parse_args()
    
    # Set low CPU priority immediately if GPU-safe mode
    if args.gpu_safe:
        set_low_priority()
        print("🔧 Set low CPU priority to avoid interfering with GPU training")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info("🧠 Starting SAFE DeepLesion Preprocessing Pipeline")
    logger.info(f"Input directory: {PNG_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Safety settings: {MAX_WORKERS} workers, {args.batch_size} batch size")
    logger.info(f"Test mode: {'ON' if args.test else 'OFF'}")
    
    # System info
    memory_info = psutil.virtual_memory()
    logger.info(f"System: {memory_info.total / (1024**3):.1f}GB RAM, {memory_info.percent:.1f}% used")
    
    # Step 1: Load metadata
    try:
        df_meta = pd.read_csv(METADATA_CSV)
        logger.info(f"Loaded metadata for {len(df_meta)} entries")
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {METADATA_CSV}")
        return
    
    # Create series UID mapping
    df_meta['series_uid'] = df_meta.apply(
        lambda row: f"{row['Patient_index']:06d}_{row['Study_index']:02d}_{row['Series_ID']:02d}",
        axis=1
    )
    spacing_map = df_meta.set_index('series_uid')['Spacing_mm_px_'].to_dict()
    
    # Step 2: Group PNG files by series
    logger.info("Scanning PNG files...")
    all_png_files = list(PNG_DIR.rglob("*.png"))
    
    series_files = defaultdict(list)
    for f in tqdm(all_png_files, desc="Grouping"):
        series_uid = f.parent.name
        series_files[series_uid].append(f)
    
    logger.info(f"Found {len(all_png_files)} PNG files in {len(series_files)} series")
    
    # Step 3: Validate and prepare tasks
    already_processed = get_already_processed(OUTPUT_DIR)
    logger.info(f"Found {len(already_processed)} already processed series")
    
    valid_tasks = []
    skipped_count = 0
    
    for series_uid, slice_paths in tqdm(series_files.items(), desc="Validating"):
        if series_uid in already_processed:
            skipped_count += 1
            continue
            
        if series_uid not in spacing_map:
            continue
            
        # Validate series
        is_valid, message = validate_series_files(series_uid, slice_paths, logger)
        if is_valid:
            valid_tasks.append((
                series_uid,
                slice_paths,
                spacing_map[series_uid],
                str(OUTPUT_DIR)
            ))
    
    logger.info(f"Prepared {len(valid_tasks)} valid tasks ({skipped_count} already processed)")
    
    # Test mode limitation
    if args.test:
        valid_tasks = valid_tasks[:10]  # Very small test
        logger.info(f"Test mode: Limited to {len(valid_tasks)} tasks")
    
    if not valid_tasks:
        logger.info("No tasks to process. Exiting.")
        return
    
    # Step 4: Process in batches
    batch_size = args.batch_size
    total_batches = (len(valid_tasks) + batch_size - 1) // batch_size
    all_results = []
    
    logger.info(f"Starting batch processing: {total_batches} batches of ~{batch_size} series each")
    
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, len(valid_tasks))
        batch_tasks = valid_tasks[start_idx:end_idx]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH {batch_num}/{total_batches}")
        logger.info(f"{'='*60}")
        
        batch_results = process_batch(batch_tasks, batch_num, total_batches, logger)
        all_results.extend(batch_results)
        
        # Save intermediate results
        if batch_results:
            intermediate_df = pd.DataFrame(all_results)
            intermediate_log = OUTPUT_DIR / f"intermediate_log_batch_{batch_num}.csv"
            intermediate_df.to_csv(intermediate_log, index=False)
            
            # Status summary
            success_count = sum(1 for r in all_results if r["status"] == "SUCCESS")
            logger.info(f"Progress: {len(all_results)}/{len(valid_tasks)} tasks, {success_count} successful")
        
        # Health check between batches
        healthy, status = check_system_health()
        logger.info(f"Post-batch health: {status}")
        if not healthy:
            logger.warning("System health degraded, stopping processing")
            break
        
        # Inter-batch cleanup
        time.sleep(2)  # Brief pause
        gc.collect()
    
    # Step 5: Final report
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    
    if all_results:
        df_log = pd.DataFrame(all_results)
        final_log = OUTPUT_DIR / "deeplesion_safe_processing_log.csv"
        df_log.to_csv(final_log, index=False)
        
        # Statistics
        status_counts = df_log['status'].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
        
        successful = df_log[df_log['status'] == 'SUCCESS']
        if not successful.empty:
            avg_time = successful['processing_time'].mean()
            avg_memory = successful['memory_peak'].mean()
            logger.info(f"Performance: {avg_time:.1f}s avg, {avg_memory:.1f}MB peak memory")
            
            # Calculate output size
            try:
                total_size = sum(f.stat().st_size for f in (OUTPUT_DIR / "CT").glob("*.nii.gz"))
                logger.info(f"Output size: {total_size / (1024**3):.2f} GB")
            except:
                logger.info("Output size: Could not calculate")
        
        # Failed cases
        failed = df_log[df_log['status'] == 'FAILED']
        if not failed.empty:
            logger.warning(f"Failed cases: {len(failed)}")
            failure_reasons = failed['message'].value_counts()
            for reason, count in failure_reasons.head(5).items():
                logger.warning(f"  '{reason}': {count} cases")
        
        logger.info(f"Detailed log: {final_log}")
    else:
        logger.error("No results collected")
    
    logger.info("🎉 SAFE DeepLesion preprocessing completed!")

if __name__ == "__main__":
    main()