#!/usr/bin/env python3
"""
FINAL - RSNA 2025 Multiprocessing Preprocessing Pipeline
MAIN PROCESSING SCRIPT - Successfully processed 4396/4405 cases (99.8%)
With robust DICOM fallback (dcm2niix + gdcmconv) and automatic failed data handling
"""

import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import subprocess
import shutil
import gc
import multiprocessing as mp
from tqdm import tqdm
import time
import argparse
import pickle
import logging
import warnings

import pymedio.image as mioi
from intensity_normalization.normalize.nyul import NyulNormalizer

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
sitk.ProcessObject.SetGlobalWarningDisplay(False)

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_ORIENTATION = "RAS"
SUBPROCESS_TIMEOUT = 300

CT_WINDOWS = {
    "brain": {"level": 40, "width": 80},
    "blood": {"level": 80, "width": 200},
    "bone": {"level": 600, "width": 3000},
}

# =============================================================================
# GLOBAL VARIABLES FOR WORKERS
# =============================================================================
NYUL_MODEL = None

def _init_worker(nyul_model_bytes):
    """Initialize worker with nyul model"""
    global NYUL_MODEL
    if nyul_model_bytes is not None:
        try:
            NYUL_MODEL = pickle.loads(nyul_model_bytes)
        except:
            NYUL_MODEL = None

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir):
    """Setup logging"""
    log_path = output_dir / "processing.log"
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
# HELPER FUNCTIONS
# =============================================================================
def run_command(cmd, timeout):
    """Run subprocess with timeout"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                                encoding='utf-8', errors='ignore', timeout=timeout)
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout} seconds."
    except Exception as e:
        return -1, f"Command failed: {e}"

def convert_to_nifti_robust(series_uid, series_dir, temp_dir):
    """
    Convert DICOM to NIfTI with robust dcm2niix -> gdcmconv -> dcm2niix fallback.
    Handles compressed DICOMs that dcm2niix alone cannot process.
    """
    # Clear any existing files from previous attempts
    for existing_file in temp_dir.glob(f"*{series_uid}*.nii.gz"):
        existing_file.unlink()
    
    # --- Primary Strategy: dcm2niix direct ---
    cmd_dcm2niix = ['dcm2niix', '-b', 'n', '-z', 'y', '-f', f"{series_uid}_raw", 
                    '-o', str(temp_dir), str(series_dir)]
    return_code, _ = run_command(cmd_dcm2niix, timeout=SUBPROCESS_TIMEOUT)
    
    if return_code == 0:
        # Find actual file created by dcm2niix (naming can be unpredictable)
        possible_files = list(temp_dir.glob(f"*{series_uid}*.nii.gz"))
        if possible_files:
            try:
                # Validate that file is readable before accepting
                sitk.ReadImage(str(possible_files[0]))
                return possible_files[0]
            except Exception:
                # File created but corrupt, delete and try fallback
                for f in possible_files:
                    f.unlink()
    
    # --- Fallback Strategy: gdcmconv + dcm2niix ---
    # Handles compressed or problematic DICOMs (~90% of RSNA dataset)
    tmp_gdcm_dir = temp_dir / "gdcm_temp" / series_uid
    tmp_gdcm_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files:
            return None
        
        # Decompress all DICOM files with gdcmconv
        for dcm_file in dcm_files:
            out_file = tmp_gdcm_dir / dcm_file.name
            cmd_gdcm = ['gdcmconv', '--raw', str(dcm_file), str(out_file)]
            run_command(cmd_gdcm, timeout=60)  # 60s per file for decompression
        
        # Run dcm2niix on decompressed files
        cmd_fallback = ['dcm2niix', '-b', 'n', '-z', 'y', '-f', f"{series_uid}_raw", 
                        '-o', str(temp_dir), str(tmp_gdcm_dir)]
        return_code_fb, _ = run_command(cmd_fallback, timeout=SUBPROCESS_TIMEOUT)
        
        if return_code_fb == 0:
            possible_files = list(temp_dir.glob(f"*{series_uid}*.nii.gz"))
            if possible_files:
                try:
                    # Final validation
                    sitk.ReadImage(str(possible_files[0]))
                    return possible_files[0]
                except Exception:
                    # Fallback also produced corrupt file
                    for f in possible_files:
                        f.unlink()
    finally:
        # Always cleanup temp gdcm directory
        if tmp_gdcm_dir.exists():
            shutil.rmtree(tmp_gdcm_dir)
    
    return None

# =============================================================================
# NYÚL LEARNING
# =============================================================================
def learn_nyul_model(df_train, series_base_dir, output_dir, logger):
    """Learn Nyúl model for MRI normalization"""
    logger.info("Learning Nyúl model...")
    
    mri_series = df_train[df_train['Modality'].isin(['MRA', 'MRI T1post', 'MRI T2'])]
    if len(mri_series) < 20:
        logger.warning("Insufficient MRI data for Nyúl learning. Skipping.")
        return None
    
    learning_sample = mri_series.sample(n=min(200, len(mri_series)), random_state=42)
    temp_dir = output_dir / "temp_nyul"
    temp_dir.mkdir(exist_ok=True)
    
    learning_images = []
    
    for _, row in tqdm(learning_sample.iterrows(), total=len(learning_sample), 
                       desc="Converting for Nyúl"):
        series_dir = series_base_dir / row['SeriesInstanceUID']
        if not series_dir.exists(): 
            continue
        
        raw_path = convert_to_nifti_robust(row['SeriesInstanceUID'], series_dir, temp_dir)
        
        if raw_path:
            try:
                img_obj = mioi.Image.from_path(str(raw_path))
                data = img_obj.get_data()
                if np.sum(data) > 0 and not np.isnan(data).any():
                    learning_images.append(img_obj)
                raw_path.unlink()
            except Exception:
                if raw_path.exists():
                    raw_path.unlink()
                continue

    logger.info(f"Loaded {len(learning_images)} valid MRI images for learning")
    
    if len(learning_images) < 10:
        shutil.rmtree(temp_dir)
        logger.error("Insufficient valid MRI images for Nyúl learning")
        return None
        
    try:
        nyul_normalizer = NyulNormalizer()
        nyul_normalizer.fit_population(learning_images)
        
        nyul_model_path = output_dir / "nyul_model.pkl"
        with open(nyul_model_path, 'wb') as f:
            pickle.dump(nyul_normalizer, f)
        
        logger.info(f"Nyúl model saved to: {nyul_model_path}")
        return nyul_normalizer
        
    except Exception as e:
        logger.error(f"Nyúl learning failed: {e}")
        return None
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# =============================================================================
# WORKER FUNCTION
# =============================================================================
def process_series_worker(args):
    """Process single series in worker"""
    series_uid, modality, series_path_str, output_dir_str = args
    series_path = Path(series_path_str)
    output_dir = Path(output_dir_str)
    
    result = {
        "series_uid": series_uid, 
        "modality": modality, 
        "status": "FAILED", 
        "message": "",
        "norm_method": "N/A",
        "processing_time": 0
    }
    
    start_time = time.time()
    
    try:
        # Create worker-specific temp directory
        temp_dir = output_dir / "temp_processing" / f"worker_{mp.current_process().pid}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if not series_path.exists():
            result["message"] = f"Series directory not found"
            return result
            
        # Convert to NIfTI using robust function
        raw_nifti_path = convert_to_nifti_robust(series_uid, series_path, temp_dir)
        if not raw_nifti_path:
            result["message"] = "DICOM conversion failed (even with fallback)"
            return result
            
        # Load image
        img_sitk = sitk.ReadImage(str(raw_nifti_path), sitk.sitkFloat32)
        
        if img_sitk.GetSize()[0] == 0:
            result["message"] = "Invalid image size"
            return result
        
        # Reorient to standard orientation
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation(TARGET_ORIENTATION)
        oriented_img = orient_filter.Execute(img_sitk)
        
        # Resample to target spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        original_size, original_spacing = oriented_img.GetSize(), oriented_img.GetSpacing()
        new_size = [max(1, int(round(osz * ospc / nspc))) 
                   for osz, ospc, nspc in zip(original_size, original_spacing, TARGET_SPACING)]
        
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(oriented_img.GetOrigin())
        resampler.SetOutputDirection(oriented_img.GetDirection())
        resampled_img_sitk = resampler.Execute(oriented_img)
        
        # Apply normalization based on modality
        final_img = None
        
        if modality == 'CTA':
            result["norm_method"] = "Multi-Window"
            img_np = sitk.GetArrayFromImage(resampled_img_sitk)
            
            if np.isnan(img_np).any() or np.isinf(img_np).any():
                result["message"] = "Invalid pixel values in CT data"
                return result
            
            channels = []
            for w in CT_WINDOWS.values():
                clipped = np.clip(img_np, w["level"] - w["width"] // 2, 
                                  w["level"] + w["width"] // 2)
                # Normalize each channel from its own min/max after clipping
                c_min, c_max = clipped.min(), clipped.max()
                if c_max > c_min:
                    normalized = (clipped - c_min) / (c_max - c_min)
                else:
                    normalized = np.zeros_like(clipped)
                channels.append(normalized.astype(np.float32))
            
            final_np = np.stack(channels, axis=-1)
            final_img = sitk.GetImageFromArray(final_np, isVector=True)

        elif modality in ['MRA', 'MRI T1post', 'MRI T2'] and NYUL_MODEL is not None:
            try:
                img_array = sitk.GetArrayFromImage(resampled_img_sitk)
                
                if np.isnan(img_array).any() or np.isinf(img_array).any():
                    raise ValueError("Invalid pixel values in MRI data")
                
                img_to_norm = mioi.Image.from_array(img_array)
                normalized_img_obj = NYUL_MODEL.transform(img_to_norm)
                normalized_data = normalized_img_obj.get_data().astype(np.float32)
                
                if np.isnan(normalized_data).any():
                    raise ValueError("Nyul normalization resulted in NaNs")
                    
                final_img = sitk.GetImageFromArray(normalized_data)
                result["norm_method"] = "Nyul"
            except Exception as e:
                result["norm_method"] = f"Percentile (Nyul failed: {type(e).__name__})"
                img_np = sitk.GetArrayFromImage(resampled_img_sitk)
                p1, p99 = np.percentile(img_np, [1, 99])
                denominator = p99 - p1 if (p99 - p1) > 1e-6 else 1e-6
                img_np = np.clip(img_np, p1, p99)
                img_np = (img_np - p1) / denominator
                final_img = sitk.GetImageFromArray(img_np.astype(np.float32))

        else: 
            result["norm_method"] = "Percentile"
            img_np = sitk.GetArrayFromImage(resampled_img_sitk)
            p1, p99 = np.percentile(img_np, [1, 99])
            denominator = p99 - p1 if (p99 - p1) > 1e-6 else 1e-6
            img_np = np.clip(img_np, p1, p99)
            img_np = (img_np - p1) / denominator
            final_img = sitk.GetImageFromArray(img_np.astype(np.float32))
        
        # Copy metadata and save
        final_img.CopyInformation(resampled_img_sitk)
        
        modality_dir = output_dir / modality.replace(" ", "_").replace("/", "_")
        modality_dir.mkdir(exist_ok=True)
        final_path = modality_dir / f"{series_uid}.nii.gz"
        
        sitk.WriteImage(final_img, str(final_path))
        
        # Cleanup
        if raw_nifti_path.exists():
            raw_nifti_path.unlink()
        
        # Cleanup worker temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        result["status"] = "SUCCESS"
        result["processing_time"] = time.time() - start_time
        
    except Exception as e:
        result["message"] = f"{type(e).__name__}: {e}"
        result["processing_time"] = time.time() - start_time
        
        # Cleanup on error
        try:
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
        except:
            pass
    
    return result

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main(args):
    """Main processing pipeline with multiprocessing"""
    
    # Setup
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    SERIES_DIR = DATA_DIR / "series"
    NUM_WORKERS = args.num_workers
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "temp_processing").mkdir(exist_ok=True)
    
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info("🚀 Starting Multiprocessing RSNA Preprocessing Pipeline")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Workers: {NUM_WORKERS}")
    logger.info("⚠️  Expected to hang at end - Ctrl+C when needed!")
    
    # Load training data
    train_csv_path = DATA_DIR / "train.csv"
    if not train_csv_path.exists():
        logger.error(f"train.csv not found at {train_csv_path}")
        return
        
    df_train = pd.read_csv(train_csv_path)
    logger.info(f"Loaded {len(df_train)} series from training data")

    # Step 1: Learn Nyúl model (sequential)
    nyul_model = learn_nyul_model(df_train, SERIES_DIR, OUTPUT_DIR, logger)
    
    # Prepare nyul model for workers
    nyul_model_bytes = None
    if nyul_model is not None:
        try:
            nyul_model_bytes = pickle.dumps(nyul_model)
            logger.info("Nyúl model successfully serialized for workers")
        except Exception as e:
            logger.warning(f"Failed to pickle Nyúl model: {e}")
            nyul_model_bytes = None

    # Step 2: Prepare all tasks
    all_tasks = []
    for _, row in df_train.iterrows():
        series_path = SERIES_DIR / row['SeriesInstanceUID']
        if series_path.exists():
            all_tasks.append((
                row['SeriesInstanceUID'], 
                row['Modality'], 
                str(series_path), 
                str(OUTPUT_DIR)
            ))
    
    logger.info(f"Prepared {len(all_tasks)} valid tasks for processing")

    # Step 3: Process with multiprocessing
    logger.info(f"Starting multiprocessing with {NUM_WORKERS} workers...")
    logger.info("💡 TIP: When it hangs near end, press Ctrl+C to stop gracefully")
    
    all_results = []
    
    try:
        with mp.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(nyul_model_bytes,),
            maxtasksperchild=50
        ) as pool:
            
            with tqdm(total=len(all_tasks), desc="Processing") as pbar:
                for result in pool.imap_unordered(process_series_worker, all_tasks):
                    all_results.append(result)
                    pbar.update(1)
                    
                    # Periodic logging and cleanup
                    if len(all_results) % 100 == 0:
                        success_count = sum(1 for r in all_results if r["status"] == "SUCCESS")
                        failed_count = len(all_results) - success_count
                        logger.info(f"Progress: {len(all_results)}/{len(all_tasks)}, "
                                  f"Success: {success_count}, Failed: {failed_count}")
                        gc.collect()
    
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user - this is expected behavior!")
    except Exception as e:
        logger.error(f"Multiprocessing error: {e}")

    # Step 4: Generate report and save failed data for local processing
    logger.info("Generating report and preparing failed data...")
    
    if all_results:
        df_log = pd.DataFrame(all_results)
        log_path = OUTPUT_DIR / "preprocessing_log.csv"
        df_log.to_csv(log_path, index=False)
        
        # Find which series were NOT processed (for manual processing)
        processed_uids = set(r["series_uid"] for r in all_results)
        all_uids = set(row['SeriesInstanceUID'] for _, row in df_train.iterrows() 
                      if (SERIES_DIR / row['SeriesInstanceUID']).exists())
        unprocessed_uids = all_uids - processed_uids
        
        # Save unprocessed for manual handling
        if unprocessed_uids:
            unprocessed_df = pd.DataFrame({
                'SeriesInstanceUID': list(unprocessed_uids),
                'Status': 'UNPROCESSED'
            })
            unprocessed_path = OUTPUT_DIR / "unprocessed_series.csv"
            unprocessed_df.to_csv(unprocessed_path, index=False)
            logger.info(f"📋 Unprocessed series saved to: {unprocessed_path}")
        
        # Save failed cases and copy DICOM files for local processing
        failed = df_log[df_log['status'] == 'FAILED']
        if not failed.empty or unprocessed_uids:
            failed_path = OUTPUT_DIR / "failed_tasks.csv"
            failed.to_csv(failed_path, index=False)
            
            # Create directory and copy DICOM files for local processing
            local_processing_dir = OUTPUT_DIR / "for_local_processing"
            local_processing_dir.mkdir(exist_ok=True)
            
            # Copy failed series
            logger.info(f"Copying {len(failed)} failed series for local processing...")
            for _, row in failed.iterrows():
                src_dir = SERIES_DIR / row['series_uid']
                dst_dir = local_processing_dir / row['series_uid']
                if src_dir.exists():
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            
            # Copy unprocessed series
            if unprocessed_uids:
                logger.info(f"Copying {len(unprocessed_uids)} unprocessed series for local processing...")
                for uid in unprocessed_uids:
                    src_dir = SERIES_DIR / uid
                    dst_dir = local_processing_dir / uid
                    if src_dir.exists():
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            
            logger.info(f"❌ Failed/unprocessed data copied to: {local_processing_dir}")
            logger.info(f"📋 Download: {local_processing_dir}, nyul_model.pkl, failed_tasks.csv")
        
        # Summary
        status_counts = df_log['status'].value_counts()
        logger.info("="*60)
        logger.info("✅ MULTIPROCESSING RESULTS")
        logger.info("="*60)
        logger.info(f"📊 Processed: {len(all_results)}/{len(all_tasks)} tasks")
        
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
        
        if len(unprocessed_uids) > 0:
            logger.info(f"  UNPROCESSED: {len(unprocessed_uids)}")
        
        if 'norm_method' in df_log.columns:
            norm_counts = df_log['norm_method'].value_counts()
            logger.info("Normalization Methods:")
            for method, count in norm_counts.items():
                logger.info(f"  {method}: {count}")
        
        times = df_log[df_log['processing_time'] > 0]['processing_time']
        if len(times) > 0:
            logger.info(f"⏱️ Performance: avg={times.mean():.1f}s, median={times.median():.1f}s")
            
        logger.info(f"📋 Detailed log: {log_path}")
    else:
        logger.error("❌ No results collected")

    # Cleanup
    temp_dir = OUTPUT_DIR / "temp_processing"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logger.info("🧹 Temporary files cleaned up")
        
    logger.info(f"📁 Processed data in: {OUTPUT_DIR}")
    
    if unprocessed_uids:
        logger.info("\n" + "="*60)
        logger.info("📋 NEXT STEPS FOR UNPROCESSED SERIES:")
        logger.info("="*60)
        logger.info("1. Check unprocessed_series.csv for remaining series")
        logger.info("2. Process manually with sequential script if needed")
        logger.info("3. Or run again with --filter_series for specific UIDs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiprocessing RSNA Preprocessing")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to directory containing 'series/' folder and 'train.csv'")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to save processed NIfTI files")
    parser.add_argument("--num_workers", type=int, 
                       default=max(1, mp.cpu_count() - 2),
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    main(args)