#!/usr/bin/env python3
"""
Complete OpenMind Preprocessing Pipeline for RSNA 2025
Based on RSNA preprocessing pattern with robust fallbacks
"""

import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import time
import pickle
import logging
import warnings
import gc
import shutil

import pymedio.image as mioi
from intensity_normalization.normalize.nyul import NyulNormalizer

# Suppress warnings like RSNA pipeline
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sitk.ProcessObject.SetGlobalWarningDisplay(False)

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_ORIENTATION = "RAS"
OUTPUT_DIR = Path("OpenMind_processed")

# Modality mapping to RSNA equivalents
MODALITY_MAPPING = {
    'T1w': 'MRI_T1',
    'T2w': 'MRI_T2', 
    'FLAIR': 'MRI_T2',
    'MP2RAGE': 'MRI_T1',
    'angio': 'MRA'
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
def setup_logging():
    """Setup logging"""
    log_path = "openmind_processing.log"
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
# NYÚL LEARNING
# =============================================================================
def learn_nyul_model(df_keep, logger):
    """Learn Nyúl model for MRI normalization"""
    logger.info("Learning Nyúl model...")
    
    mri_series = df_keep[df_keep['modality'].isin(['T1w', 'T2w', 'FLAIR', 'MP2RAGE'])]
    if len(mri_series) < 20:
        logger.warning("Insufficient MRI data for Nyúl learning. Skipping.")
        return None
    
    learning_sample = mri_series.sample(n=min(200, len(mri_series)), random_state=42)
    learning_images = []
    
    for _, row in tqdm(learning_sample.iterrows(), total=len(learning_sample), 
                       desc="Loading for Nyúl"):
        file_path = Path("OpenMind") / row['image_path']
        if not file_path.exists(): 
            continue
        
        try:
            img_sitk = sitk.ReadImage(str(file_path), sitk.sitkFloat32)
            img_array = sitk.GetArrayFromImage(img_sitk)
            
            if (np.sum(img_array) > 0 and 
                not np.isnan(img_array).any() and 
                not np.isinf(img_array).any()):
                
                img_obj = mioi.Image.from_array(img_array)
                learning_images.append(img_obj)
                
        except Exception:
            continue

    logger.info(f"Loaded {len(learning_images)} valid MRI images for learning")
    
    if len(learning_images) < 10:
        logger.error("Insufficient valid MRI images for Nyúl learning")
        return None
        
    try:
        nyul_normalizer = NyulNormalizer()
        nyul_normalizer.fit_population(learning_images)
        
        nyul_model_path = "nyul_model.pkl"
        with open(nyul_model_path, 'wb') as f:
            pickle.dump(nyul_normalizer, f)
        
        logger.info(f"Nyúl model saved to: {nyul_model_path}")
        return nyul_normalizer
        
    except Exception as e:
        logger.error(f"Nyúl learning failed: {e}")
        return None

# =============================================================================
# WORKER FUNCTION
# =============================================================================
def process_file_worker(args):
    """Process single file with robust error handling like RSNA pipeline"""
    image_path, modality, output_dir_str = args
    output_dir = Path(output_dir_str)
    
    # Extract ID from path for logging
    file_id = Path(image_path).stem
    
    result = {
        "image_path": image_path,
        "file_id": file_id,
        "modality": modality,
        "mapped_modality": MODALITY_MAPPING.get(modality, modality),
        "status": "FAILED",
        "message": "",
        "norm_method": "N/A",
        "processing_time": 0
    }
    
    start_time = time.time()
    
    try:
        file_path = Path("OpenMind") / image_path
        if not file_path.exists():
            result["message"] = f"File not found"
            return result
            
        # Load image with validation
        img_sitk = sitk.ReadImage(str(file_path), sitk.sitkFloat32)
        
        if img_sitk.GetSize()[0] == 0:
            result["message"] = "Invalid image size"
            return result
        
        # Early validation like RSNA pipeline
        img_array = sitk.GetArrayFromImage(img_sitk)
        if np.isnan(img_array).any() or np.isinf(img_array).any():
            result["message"] = "Invalid pixel values (NaN/Inf)"
            return result
            
        if np.sum(img_array) == 0:
            result["message"] = "Empty image (all zeros)"
            return result
        
        # Step 1: Reorient to standard orientation
        try:
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(TARGET_ORIENTATION)
            oriented_img = orient_filter.Execute(img_sitk)
        except Exception:
            oriented_img = img_sitk  # Use original if orientation fails
        
        # Step 2: Resample to target spacing
        try:
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(TARGET_SPACING)
            resampler.SetInterpolator(sitk.sitkLinear)
            
            original_size, original_spacing = oriented_img.GetSize(), oriented_img.GetSpacing()
            new_size = [max(1, int(round(osz * ospc / nspc))) 
                       for osz, ospc, nspc in zip(original_size, original_spacing, TARGET_SPACING)]
            
            # Safety check for reasonable dimensions
            new_size = [min(dim, 1024) for dim in new_size]
            
            resampler.SetSize(new_size)
            resampler.SetOutputOrigin(oriented_img.GetOrigin())
            resampler.SetOutputDirection(oriented_img.GetDirection())
            resampled_img_sitk = resampler.Execute(oriented_img)
            
        except Exception:
            resampled_img_sitk = oriented_img  # Use oriented if resampling fails
        
        # Step 3: Apply normalization based on modality
        mapped_modality = MODALITY_MAPPING.get(modality, modality)
        final_img = None
        
        if mapped_modality in ['MRI_T1', 'MRI_T2', 'MRA'] and NYUL_MODEL is not None:
            try:
                img_array = sitk.GetArrayFromImage(resampled_img_sitk)
                
                # Validate for Nyúl processing
                if (not np.isnan(img_array).any() and 
                    not np.isinf(img_array).any() and
                    img_array.max() > img_array.min()):
                    
                    img_to_norm = mioi.Image.from_array(img_array)
                    normalized_img_obj = NYUL_MODEL.transform(img_to_norm)
                    normalized_data = normalized_img_obj.get_data().astype(np.float32)
                    
                    if not np.isnan(normalized_data).any():
                        final_img = sitk.GetImageFromArray(normalized_data)
                        result["norm_method"] = "Nyul"
                    else:
                        raise ValueError("Nyul normalization resulted in NaNs")
                else:
                    raise ValueError("Invalid array for Nyul processing")
                    
            except Exception as e:
                # Fallback to percentile normalization
                result["norm_method"] = f"Percentile (Nyul failed: {type(e).__name__})"
                img_np = sitk.GetArrayFromImage(resampled_img_sitk)
                p1, p99 = np.percentile(img_np, [1, 99])
                denominator = p99 - p1 if (p99 - p1) > 1e-6 else 1e-6
                img_np = np.clip(img_np, p1, p99)
                img_np = (img_np - p1) / denominator
                final_img = sitk.GetImageFromArray(img_np.astype(np.float32))

        else: 
            # Direct percentile normalization
            result["norm_method"] = "Percentile"
            img_np = sitk.GetArrayFromImage(resampled_img_sitk)
            p1, p99 = np.percentile(img_np, [1, 99])
            denominator = p99 - p1 if (p99 - p1) > 1e-6 else 1e-6
            img_np = np.clip(img_np, p1, p99)
            img_np = (img_np - p1) / denominator
            final_img = sitk.GetImageFromArray(img_np.astype(np.float32))
        
        # Copy metadata and save
        final_img.CopyInformation(resampled_img_sitk)
        
        modality_dir = output_dir / mapped_modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        final_path = modality_dir / f"{file_id}.nii.gz"
        
        sitk.WriteImage(final_img, str(final_path))
        
        result["status"] = "SUCCESS"
        result["processing_time"] = time.time() - start_time
        
    except Exception as e:
        result["message"] = f"{type(e).__name__}: {e}"
        result["processing_time"] = time.time() - start_time
    
    return result

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main processing pipeline"""
    
    logger = setup_logging()
    
    logger.info("🧠 Starting OpenMind Preprocessing Pipeline")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Load target files
    df_keep = pd.read_csv('keep_files_validated.csv')
    logger.info(f"Loaded {len(df_keep)} files to process")

    # Show modality breakdown
    modality_counts = df_keep['modality'].value_counts()
    logger.info("Modality breakdown:")
    for modality, count in modality_counts.items():
        mapped = MODALITY_MAPPING.get(modality, modality)
        logger.info(f"  {modality} → {mapped}: {count}")
    
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Learn Nyúl model
    nyul_model = learn_nyul_model(df_keep, logger)
    
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
    for _, row in df_keep.iterrows():
        all_tasks.append((
            row['image_path'],
            row['modality'],
            str(OUTPUT_DIR)
        ))
    
    logger.info(f"Prepared {len(all_tasks)} processing tasks")

    # Step 3: Process with multiprocessing
    NUM_WORKERS = 14
    logger.info(f"Starting multiprocessing with {NUM_WORKERS} workers...")
    
    all_results = []
    
    try:
        with mp.Pool(
            processes=NUM_WORKERS,
            initializer=_init_worker,
            initargs=(nyul_model_bytes,),
            maxtasksperchild=50
        ) as pool:
            
            with tqdm(total=len(all_tasks), desc="Processing") as pbar:
                for result in pool.imap_unordered(process_file_worker, all_tasks):
                    all_results.append(result)
                    pbar.update(1)
                    
                    # Periodic logging and cleanup
                    if len(all_results) % 1000 == 0:
                        success_count = sum(1 for r in all_results if r["status"] == "SUCCESS")
                        failed_count = len(all_results) - success_count
                        logger.info(f"Progress: {len(all_results)}/{len(all_tasks)}, "
                                  f"Success: {success_count}, Failed: {failed_count}")
                        gc.collect()
    
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"Multiprocessing error: {e}")

    # Step 4: Generate report
    logger.info("Generating results report...")
    
    if all_results:
        df_log = pd.DataFrame(all_results)
        log_path = OUTPUT_DIR / "processing_log.csv"
        df_log.to_csv(log_path, index=False)
        
        # Summary statistics
        status_counts = df_log['status'].value_counts()
        logger.info("=" * 60)
        logger.info("✅ PROCESSING RESULTS")
        logger.info("=" * 60)
        logger.info(f"📊 Processed: {len(all_results)}/{len(all_tasks)} tasks")
        
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
        
        if 'norm_method' in df_log.columns:
            norm_counts = df_log['norm_method'].value_counts()
            logger.info("Normalization Methods:")
            for method, count in norm_counts.items():
                logger.info(f"  {method}: {count}")
        
        # Performance stats
        times = df_log[df_log['processing_time'] > 0]['processing_time']
        if len(times) > 0:
            logger.info(f"⏱️ Performance: avg={times.mean():.1f}s, median={times.median():.1f}s")
        
        # Output breakdown
        success_df = df_log[df_log['status'] == 'SUCCESS']
        if not success_df.empty:
            mapped_counts = success_df['mapped_modality'].value_counts()
            logger.info("Output by modality:")
            for modality, count in mapped_counts.items():
                logger.info(f"  {modality}: {count}")
        
        # Calculate final size
        total_size_gb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*.nii.gz')) / (1024**3)
        logger.info(f"💾 Final size: {total_size_gb:.1f}GB")
        
        logger.info(f"📋 Detailed log: {log_path}")
    else:
        logger.error("❌ No results collected")
    
    logger.info("🎉 Processing completed!")
    logger.info(f"📁 Processed data: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()