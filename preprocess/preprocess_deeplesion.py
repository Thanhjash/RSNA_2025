#!/usr/bin/env python3
"""
DeepLesion Preprocessing Pipeline for RSNA 2025

This script processes the DeepLesion dataset, which consists of 16-bit PNG slices,
and harmonizes it with the output of the RSNA and OpenMind pipelines.

Key Steps:
1. Reads the DL_info.csv to get metadata, especially image spacing.
2. Groups individual PNG slices into 3D series based on their filenames.
3. For each series:
    a. Loads the 16-bit PNG slices and stacks them into a 3D volume.
    b. **Crucially, converts pixel intensities to Hounsfield Units (HU) by subtracting 32768.**
    c. Creates a SimpleITK image, setting the initial voxel spacing from the metadata.
    d. Reorients the image to our standard 'RAS' orientation.
    e. Resamples the image to isotropic 1.0x1.0x1.0 mm spacing.
    f. Applies the exact same 3-channel CT windowing (brain, blood, bone) as the RSNA pipeline.
    g. Saves the final 3-channel image as a .nii.gz file.
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

# Suppress warnings, consistent with other pipelines
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sitk.ProcessObject.SetGlobalWarningDisplay(False)

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- Input Paths ---
# Assumes the script is run from the project root (e.g., /mnt/d/2.Research/RSNA)
DEEPLESION_RAW_DIR = Path("data/raw/NIH_deeplesion")
PNG_DIR = DEEPLESION_RAW_DIR / "minideeplesion"
METADATA_CSV = DEEPLESION_RAW_DIR / "DL_info.csv"

# --- Output Path ---
OUTPUT_DIR = Path("data/processed/NIH_deeplesion")

# --- Harmonization Settings (MUST MATCH RSNA_data_processor.py) ---
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_ORIENTATION = "RAS"
CT_WINDOWS = {
    "brain": {"level": 40, "width": 80},
    "blood": {"level": 80, "width": 200},
    "bone": {"level": 600, "width": 3000},
}

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir):
    """Sets up logging to file and console."""
    log_path = output_dir / "deeplesion_processing.log"
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
# WORKER FUNCTION
# =============================================================================
def process_series_worker(args):
    """
    Processes one full 3D series from a list of 2D PNG slices.
    """
    series_uid, slice_paths, spacing_info, output_dir = args
    
    result = {
        "series_uid": series_uid,
        "status": "FAILED",
        "message": "",
        "processing_time": 0
    }
    start_time = time.time()

    try:
        # Step 1: Load, sort, and stack slices
        slices = []
        slice_indices = []
        for slice_path in slice_paths:
            try:
                slice_idx = int(slice_path.stem.split('_')[-1])
                slice_indices.append(slice_idx)
                slices.append(imageio.imread(slice_path))
            except Exception as e:
                result["message"] = f"Failed to read slice {slice_path.name}: {e}"
                return result

        # Sort slices by index to ensure correct order
        sorted_indices = np.argsort(slice_indices)
        sorted_slices = [slices[i] for i in sorted_indices]
        
        # Stack into a 3D volume
        volume_np = np.stack(sorted_slices, axis=0)

        # Step 2: Convert to Hounsfield Units (HU)
        # This is the critical transformation for DeepLesion PNGs
        volume_np = volume_np.astype(np.float32) - 32768.0

        # Step 3: Create SimpleITK image with correct initial spacing
        img_sitk = sitk.GetImageFromArray(volume_np)
        
        # Spacing is [slice_thickness, pixel_spacing_x, pixel_spacing_y]
        # We need to parse it and reorder for SimpleITK [x, y, z]
        try:
            spacing_parts = [float(s) for s in spacing_info.split(',')]
            # Assuming CSV is [x, y, z]
            initial_spacing = (spacing_parts[0], spacing_parts[1], spacing_parts[2])
            img_sitk.SetSpacing(initial_spacing)
        except Exception as e:
            result["message"] = f"Invalid spacing format '{spacing_info}': {e}"
            return result

        # Step 4: Reorient to our standard 'RAS'
        try:
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(TARGET_ORIENTATION)
            oriented_img = orient_filter.Execute(img_sitk)
        except Exception:
            oriented_img = img_sitk # Fallback to original if orientation fails

        # Step 5: Resample to target isotropic spacing
        try:
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(TARGET_SPACING)
            resampler.SetInterpolator(sitk.sitkLinear)
            
            original_size = oriented_img.GetSize()
            original_spacing = oriented_img.GetSpacing()
            new_size = [
                max(1, int(round(osz * ospc / nspc)))
                for osz, ospc, nspc in zip(original_size, original_spacing, TARGET_SPACING)
            ]
            
            resampler.SetSize(new_size)
            resampler.SetOutputOrigin(oriented_img.GetOrigin())
            resampler.SetOutputDirection(oriented_img.GetDirection())
            resampled_img_sitk = resampler.Execute(oriented_img)
        except Exception as e:
            result["message"] = f"Resampling failed: {e}"
            return result

        # Step 6: Apply the standard multi-window CT normalization
        img_np = sitk.GetArrayFromImage(resampled_img_sitk)
        
        if np.isnan(img_np).any() or np.isinf(img_np).any():
            result["message"] = "Invalid pixel values (NaN/Inf) after resampling"
            return result

        channels = []
        for w in CT_WINDOWS.values():
            clipped = np.clip(img_np, w["level"] - w["width"] // 2, w["level"] + w["width"] // 2)
            c_min, c_max = clipped.min(), clipped.max()
            if c_max > c_min:
                normalized = (clipped - c_min) / (c_max - c_min)
            else:
                normalized = np.zeros_like(clipped)
            channels.append(normalized.astype(np.float32))
        
        final_np = np.stack(channels, axis=-1) # Create a (D, H, W, C) array
        final_img = sitk.GetImageFromArray(final_np, isVector=True)

        # Step 7: Save the final processed image
        final_img.CopyInformation(resampled_img_sitk)
        
        # All DeepLesion are CT, so save them in a 'CT' subfolder for consistency
        modality_dir = output_dir / "CT"
        modality_dir.mkdir(parents=True, exist_ok=True)
        final_path = modality_dir / f"{series_uid}.nii.gz"
        
        sitk.WriteImage(final_img, str(final_path))
        
        result["status"] = "SUCCESS"
        
    except Exception as e:
        result["message"] = f"{type(e).__name__}: {e}"
    
    result["processing_time"] = time.time() - start_time
    return result

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main orchestration function."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info("🧠 Starting DeepLesion Preprocessing Pipeline")
    logger.info(f"Input PNG directory: {PNG_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Harmonizing to {TARGET_SPACING}mm spacing and '{TARGET_ORIENTATION}' orientation.")

    # Step 1: Read metadata and get spacing info
    try:
        df_meta = pd.read_csv(METADATA_CSV)
    except FileNotFoundError:
        logger.error(f"Metadata file not found at {METADATA_CSV}. Aborting.")
        return

    # Create a mapping from series_uid to spacing
    # A series can have multiple entries, but spacing should be the same.
    df_meta['series_uid'] = df_meta.apply(
        lambda row: f"{row['Patient_index']:06d}_{row['Study_index']:02d}_{row['Series_ID']:02d}",
        axis=1
    )
    spacing_map = df_meta.set_index('series_uid')['Spacing_mm_px_'].to_dict()
    logger.info(f"Loaded metadata for {len(spacing_map)} unique series from {METADATA_CSV}")

    # Step 2: Group all PNG files by their series UID
    logger.info(f"Scanning for PNG files in {PNG_DIR}...")
    all_png_files = list(PNG_DIR.rglob("*.png"))
    
    series_files = defaultdict(list)
    for f in tqdm(all_png_files, desc="Grouping slices"):
        try:
            series_uid = f.parent.name
            series_files[series_uid].append(f)
        except IndexError:
            logger.warning(f"Could not parse series UID from filename: {f.name}")
            continue
    
    logger.info(f"Found {len(all_png_files)} PNG slices grouped into {len(series_files)} series.")

    # Step 3: Prepare tasks for multiprocessing
    all_tasks = []
    for series_uid, slice_paths in series_files.items():
        if series_uid in spacing_map:
            all_tasks.append((
                series_uid,
                slice_paths,
                spacing_map[series_uid],
                OUTPUT_DIR
            ))
        else:
            logger.warning(f"Skipping series {series_uid}: No spacing info found in metadata.")
            
    logger.info(f"Prepared {len(all_tasks)} valid tasks for processing.")

    # Step 4: Run processing in parallel
    NUM_WORKERS = max(1, mp.cpu_count() - 2)
    logger.info(f"Starting multiprocessing with {NUM_WORKERS} workers...")
    
    all_results = []
    with mp.Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(all_tasks), desc="Processing Series") as pbar:
            for result in pool.imap_unordered(process_series_worker, all_tasks):
                all_results.append(result)
                pbar.update(1)

    # Step 5: Generate and log report
    logger.info("=" * 60)
    logger.info("✅ PROCESSING RESULTS")
    logger.info("=" * 60)

    if not all_results:
        logger.warning("No tasks were processed or found. Final report cannot be generated.")
        logger.info("🎉 DeepLesion processing completed (no files processed).")
        return
    
    df_log = pd.DataFrame(all_results)
    log_path = OUTPUT_DIR / "deeplesion_processing_log.csv"
    df_log.to_csv(log_path, index=False)
    
    status_counts = df_log['status'].value_counts()
    logger.info(f"📊 Processed: {len(df_log)}/{len(all_tasks)} tasks")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
        
    failed_df = df_log[df_log['status'] == 'FAILED']
    if not failed_df.empty:
        logger.warning(f"🚨 Found {len(failed_df)} failures. Check the log for details:")
        for _, row in failed_df.iterrows():
            logger.warning(f"  - Series {row['series_uid']}: {row['message']}")

    times = df_log[df_log['processing_time'] > 0]['processing_time']
    if not times.empty:
        logger.info(f"⏱️ Performance: avg={times.mean():.1f}s, median={times.median():.1f}s")

    total_size_gb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*.nii.gz')) / (1024**3)
    logger.info(f"💾 Final size of processed data: {total_size_gb:.2f} GB")
    logger.info(f"📋 Detailed log saved to: {log_path}")
    logger.info("🎉 DeepLesion processing completed!")

if __name__ == "__main__":
    main()
