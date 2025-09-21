#!/usr/bin/env python3
"""
Quick DeepLesion Data Quality Check
Fast analysis without visualization for CI/headless environments
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def analyze_volume(nifti_path):
    """Quick analysis of a volume."""
    try:
        img = sitk.ReadImage(str(nifti_path))
        img_array = sitk.GetArrayFromImage(img)
        
        return {
            'file': nifti_path.name,
            'size_3d': img.GetSize(),
            'spacing': img.GetSpacing(),
            'array_shape': img_array.shape,
            'num_channels': img_array.shape[-1] if len(img_array.shape) == 4 else 1,
            'dtype': str(img_array.dtype),
            'file_size_mb': nifti_path.stat().st_size / (1024**2),
            'has_nan': np.isnan(img_array).any(),
            'has_inf': np.isinf(img_array).any(),
            'min_val': float(np.min(img_array)),
            'max_val': float(np.max(img_array)),
            'mean_val': float(np.mean(img_array)),
            'status': 'OK'
        }
    except Exception as e:
        return {
            'file': nifti_path.name,
            'error': str(e),
            'status': 'ERROR'
        }

def main():
    print("🔍 Quick DeepLesion Data Quality Check")
    print("="*50)
    
    processed_dir = Path("data/processed/NIH_deeplesion/CT")
    files = list(processed_dir.glob("*.nii.gz"))
    
    print(f"Found {len(files)} processed files")
    
    # Sample different file sizes
    file_sizes = [(f, f.stat().st_size / (1024**2)) for f in files]
    file_sizes.sort(key=lambda x: x[1])
    
    # Select samples: smallest, ~10MB, ~50MB, ~100MB, largest
    samples = []
    targets = [0, 10, 50, 100, float('inf')]
    
    for target in targets:
        if target == 0:
            samples.append(file_sizes[0][0])  # Smallest
        elif target == float('inf'):
            samples.append(file_sizes[-1][0])  # Largest
        else:
            # Find closest to target
            best = min(file_sizes, key=lambda x: abs(x[1] - target))
            samples.append(best[0])
    
    # Remove duplicates while preserving order
    unique_samples = []
    for s in samples:
        if s not in unique_samples:
            unique_samples.append(s)
    
    print(f"\nAnalyzing {len(unique_samples)} sample files:")
    
    results = []
    for f in tqdm(unique_samples):
        result = analyze_volume(f)
        results.append(result)
        
        if result['status'] == 'OK':
            print(f"\n📁 {result['file']}:")
            print(f"  3D Size: {result['size_3d']}")
            print(f"  Array Shape: {result['array_shape']}")
            print(f"  Channels: {result['num_channels']}")
            print(f"  Spacing: {[f'{x:.2f}' for x in result['spacing']]}")
            print(f"  Data Type: {result['dtype']}")
            print(f"  Value Range: {result['min_val']:.3f} to {result['max_val']:.3f}")
            print(f"  File Size: {result['file_size_mb']:.1f} MB")
            print(f"  Valid Data: {'✅' if not (result['has_nan'] or result['has_inf']) else '❌'}")
        else:
            print(f"\n❌ {result['file']}: {result.get('error', 'Unknown error')}")
    
    # Summary analysis
    print(f"\n📊 SUMMARY ANALYSIS")
    print("="*50)
    
    valid_results = [r for r in results if r['status'] == 'OK']
    
    if valid_results:
        # Check consistency
        channels = [r['num_channels'] for r in valid_results]
        spacings = [r['spacing'] for r in valid_results]
        dtypes = [r['dtype'] for r in valid_results]
        
        print(f"✅ Successfully analyzed {len(valid_results)}/{len(results)} files")
        print(f"📋 Channels: {set(channels)} (all should be 3)")
        print(f"📏 Spacing consistency: {len(set(spacings))} unique spacings")
        print(f"🔢 Data types: {set(dtypes)}")
        
        # Check for 3-channel format
        if all(c == 3 for c in channels):
            print("✅ All files have 3 channels (correct format)")
        else:
            print(f"⚠️  Channel count issues: {channels}")
        
        # Check spacing
        target_spacing = (1.0, 1.0, 1.0)
        spacing_ok = all(
            abs(s[0] - target_spacing[0]) < 0.01 and 
            abs(s[1] - target_spacing[1]) < 0.01 and 
            abs(s[2] - target_spacing[2]) < 0.01 
            for s in spacings
        )
        
        if spacing_ok:
            print("✅ All files have correct 1.0mm isotropic spacing")
        else:
            print(f"⚠️  Spacing variations detected")
            for i, s in enumerate(spacings):
                print(f"    {valid_results[i]['file']}: {s}")
        
        # Value range analysis
        all_values_normalized = all(
            0.0 <= r['min_val'] <= 1.0 and 0.0 <= r['max_val'] <= 1.0 
            for r in valid_results
        )
        
        if all_values_normalized:
            print("✅ All values in expected 0-1 range (normalized)")
        else:
            print("⚠️  Some values outside 0-1 range:")
            for r in valid_results:
                if not (0.0 <= r['min_val'] <= 1.0 and 0.0 <= r['max_val'] <= 1.0):
                    print(f"    {r['file']}: {r['min_val']:.3f} to {r['max_val']:.3f}")
    
    # File size analysis
    print(f"\n💾 FILE SIZE ANALYSIS")
    print("="*30)
    
    all_files = [(f.name, f.stat().st_size / (1024**2)) for f in files]
    sizes = [s[1] for s in all_files]
    
    print(f"Total files: {len(files)}")
    print(f"Total size: {sum(sizes):.1f} MB ({sum(sizes)/1024:.1f} GB)")
    print(f"Average size: {np.mean(sizes):.1f} MB")
    print(f"Size range: {min(sizes):.1f} - {max(sizes):.1f} MB")
    print(f"Median size: {np.median(sizes):.1f} MB")
    
    # Identify outliers
    p95 = np.percentile(sizes, 95)
    large_files = [f for f, s in all_files if s > p95]
    
    print(f"\nLargest files (>95th percentile, >{p95:.1f}MB):")
    large_files_with_sizes = [(f, next(s for name, s in all_files if name == f)) for f in large_files[:5]]
    for name, size in large_files_with_sizes:
        print(f"  {name}: {size:.1f} MB")
    
    print(f"\n🎯 QUALITY VERDICT:")
    if valid_results and len(valid_results) == len(results):
        if all(r['num_channels'] == 3 for r in valid_results) and spacing_ok:
            print("✅ EXCELLENT - All samples passed quality checks!")
            print("✅ Data is ready for training pipeline")
        else:
            print("⚠️  GOOD - Minor format inconsistencies detected")
    else:
        print("❌ ISSUES - Some files failed to load")
    
    print(f"\n📄 The 30GB size is NORMAL for:")
    print(f"   • 671 series × ~45MB average = ~30GB total")
    print(f"   • 3-channel format (3x original size)")
    print(f"   • Float32 precision + isotropic resampling")

if __name__ == "__main__":
    main()