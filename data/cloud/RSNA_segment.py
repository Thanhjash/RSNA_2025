#!/usr/bin/env python3
"""
Process segmentation subset for RSNA 2025
Quick processing for EC2 environment
"""

import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# Match your main preprocessing config
TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_ORIENTATION = "RAS"

def setup_logging(output_dir):
    """Setup logging"""
    log_path = output_dir / "segmentation_processing.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_segmentation(seg_path, output_path, series_uid):
    """Process single segmentation file"""
    try:
        # Load segmentation
        seg_img = sitk.ReadImage(str(seg_path), sitk.sitkUInt8)
        
        if seg_img.GetSize()[0] == 0:
            return False, "Invalid image size"
        
        # Reorient to standard orientation
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation(TARGET_ORIENTATION)
        oriented_seg = orient_filter.Execute(seg_img)
        
        # Resample to target spacing (using nearest neighbor for labels)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(TARGET_SPACING)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Important for labels
        
        original_size = oriented_seg.GetSize()
        original_spacing = oriented_seg.GetSpacing()
        new_size = [max(1, int(round(osz * ospc / nspc))) 
                   for osz, ospc, nspc in zip(original_size, original_spacing, TARGET_SPACING)]
        
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(oriented_seg.GetOrigin())
        resampler.SetOutputDirection(oriented_seg.GetDirection())
        resampled_seg = resampler.Execute(oriented_seg)
        
        # Save processed segmentation
        sitk.WriteImage(resampled_seg, str(output_path))
        
        return True, "Success"
        
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main(args):
    """Main processing function"""
    
    # Setup paths
    DATA_DIR = Path(args.data_dir)
    SEGMENTATIONS_DIR = DATA_DIR / "segmentations"
    OUTPUT_DIR = Path(args.output_dir)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info("🔬 Processing RSNA Segmentation Subset")
    logger.info(f"Segmentations directory: {SEGMENTATIONS_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    if not SEGMENTATIONS_DIR.exists():
        logger.error(f"Segmentations directory not found: {SEGMENTATIONS_DIR}")
        return
    
    # Find all segmentation files
    seg_files = list(SEGMENTATIONS_DIR.glob("*.nii.gz"))
    
    if not seg_files:
        logger.error("No .nii.gz files found in segmentations directory")
        return
    
    logger.info(f"Found {len(seg_files)} segmentation files")
    
    # Process each segmentation
    results = []
    
    for seg_file in tqdm(seg_files, desc="Processing segmentations"):
        # Extract SeriesInstanceUID from filename
        series_uid = seg_file.stem.replace('.nii', '')
        
        output_path = OUTPUT_DIR / f"{series_uid}_vessel_mask.nii.gz"
        
        success, message = process_segmentation(seg_file, output_path, series_uid)
        
        result = {
            "series_uid": series_uid,
            "input_file": seg_file.name,
            "output_file": output_path.name if success else "N/A",
            "status": "SUCCESS" if success else "FAILED",
            "message": message
        }
        
        results.append(result)
        
        if len(results) % 50 == 0:
            logger.info(f"Processed {len(results)}/{len(seg_files)} files")
    
    # Save results log
    df_results = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "segmentation_processing_log.csv"
    df_results.to_csv(results_path, index=False)
    
    # Summary
    success_count = len(df_results[df_results['status'] == 'SUCCESS'])
    failed_count = len(df_results) - success_count
    
    logger.info("="*50)
    logger.info("✅ SEGMENTATION PROCESSING COMPLETE")
    logger.info("="*50)
    logger.info(f"📊 Total files: {len(seg_files)}")
    logger.info(f"✅ Success: {success_count}")
    logger.info(f"❌ Failed: {failed_count}")
    logger.info(f"📋 Detailed log: {results_path}")
    logger.info(f"📁 Processed masks: {OUTPUT_DIR}")
    
    if failed_count > 0:
        failed_files = df_results[df_results['status'] == 'FAILED']['input_file'].tolist()
        logger.warning(f"Failed files: {failed_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RSNA segmentation subset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to directory containing 'segmentations/' folder")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to save processed segmentation masks")
    
    args = parser.parse_args()
    main(args)