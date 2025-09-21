#!/usr/bin/env python3
"""
DeepLesion Data Validation Script
Quick analysis of the DeepLesion dataset without processing
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

# Paths
DEEPLESION_RAW_DIR = Path("data/raw/NIH_deeplesion")
PNG_DIR = DEEPLESION_RAW_DIR / "minideeplesion"
METADATA_CSV = DEEPLESION_RAW_DIR / "DL_info.csv"

def main():
    print("🔍 DeepLesion Dataset Validation")
    print(f"Raw directory: {DEEPLESION_RAW_DIR}")
    print(f"PNG directory: {PNG_DIR}")
    print(f"Metadata CSV: {METADATA_CSV}")
    
    # Check if directories exist
    if not PNG_DIR.exists():
        print(f"❌ PNG directory not found: {PNG_DIR}")
        return
    
    if not METADATA_CSV.exists():
        print(f"❌ Metadata CSV not found: {METADATA_CSV}")
        return
    
    # Load metadata
    print("\n📊 Loading metadata...")
    df_meta = pd.read_csv(METADATA_CSV)
    print(f"Metadata entries: {len(df_meta)}")
    
    # Create series mapping
    df_meta['series_uid'] = df_meta.apply(
        lambda row: f"{row['Patient_index']:06d}_{row['Study_index']:02d}_{row['Series_ID']:02d}",
        axis=1
    )
    
    # Analysis
    print(f"Unique patients: {df_meta['Patient_index'].nunique()}")
    print(f"Unique studies: {len(df_meta.groupby(['Patient_index', 'Study_index']))}")
    print(f"Unique series: {df_meta['series_uid'].nunique()}")
    
    # Spacing analysis
    spacing_values = df_meta['Spacing_mm_px_'].value_counts()
    print(f"\nTop 5 spacing values:")
    for spacing, count in spacing_values.head().items():
        print(f"  {spacing}: {count} cases")
    
    # Scan PNG files
    print("\n🔍 Scanning PNG files...")
    all_png_files = list(PNG_DIR.rglob("*.png"))
    print(f"Total PNG files: {len(all_png_files)}")
    
    # Group by series
    series_files = defaultdict(list)
    for f in tqdm(all_png_files, desc="Grouping"):
        series_uid = f.parent.name
        series_files[series_uid].append(f)
    
    print(f"PNG files grouped into {len(series_files)} series")
    
    # Series size analysis
    series_sizes = [len(files) for files in series_files.values()]
    series_counter = Counter(series_sizes)
    
    print(f"\nSeries size distribution:")
    print(f"  Min slices per series: {min(series_sizes)}")
    print(f"  Max slices per series: {max(series_sizes)}")
    print(f"  Average slices per series: {np.mean(series_sizes):.1f}")
    print(f"  Median slices per series: {np.median(series_sizes):.1f}")
    
    print(f"\nMost common series sizes:")
    for size, count in series_counter.most_common(10):
        print(f"  {size} slices: {count} series")
    
    # Check for missing metadata
    series_with_metadata = set(df_meta['series_uid'])
    series_with_files = set(series_files.keys())
    
    missing_metadata = series_with_files - series_with_metadata
    missing_files = series_with_metadata - series_with_files
    
    print(f"\nData consistency:")
    print(f"  Series with PNG files: {len(series_with_files)}")
    print(f"  Series with metadata: {len(series_with_metadata)}")
    print(f"  Series missing metadata: {len(missing_metadata)}")
    print(f"  Series missing files: {len(missing_files)}")
    
    # Valid series for processing
    valid_series = series_with_files & series_with_metadata
    valid_series_with_enough_slices = {
        uid for uid in valid_series 
        if len(series_files[uid]) >= 3  # Minimum for 3D
    }
    
    print(f"\nProcessing estimates:")
    print(f"  Valid series (has both files and metadata): {len(valid_series)}")
    print(f"  Valid series with ≥3 slices: {len(valid_series_with_enough_slices)}")
    
    # Estimate output size
    # Assuming ~1MB per processed series (conservative estimate)
    estimated_size_gb = len(valid_series_with_enough_slices) * 1 / 1024  # MB to GB
    print(f"  Estimated output size: {estimated_size_gb:.1f} GB")
    
    # Sample series info
    if valid_series_with_enough_slices:
        sample_uid = list(valid_series_with_enough_slices)[0]
        sample_files = series_files[sample_uid]
        sample_meta = df_meta[df_meta['series_uid'] == sample_uid].iloc[0]
        
        print(f"\nSample series: {sample_uid}")
        print(f"  Number of slices: {len(sample_files)}")
        print(f"  Spacing: {sample_meta['Spacing_mm_px_']}")
        print(f"  Patient age: {sample_meta['Patient_age']}")
        print(f"  Patient gender: {sample_meta['Patient_gender']}")
        print(f"  First slice: {sample_files[0].name}")
        print(f"  Last slice: {sample_files[-1].name}")
    
    print(f"\n✅ Validation complete!")
    print(f"Dataset is ready for processing: {len(valid_series_with_enough_slices)} series")

if __name__ == "__main__":
    main()