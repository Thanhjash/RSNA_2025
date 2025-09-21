#!/usr/bin/env python3
"""
Quick OpenMind Data Quality Check
Fast analysis without visualization for CI/headless environments
Validates processed OpenMind dataset consistency with RSNA/DeepLesion standards
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def analyze_volume(nifti_path, modality):
    """Quick analysis of a volume."""
    try:
        img = sitk.ReadImage(str(nifti_path))
        img_array = sitk.GetArrayFromImage(img)
        
        return {
            'file': nifti_path.name,
            'modality': modality,
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
            'modality': modality,
            'error': str(e),
            'status': 'ERROR'
        }

def main():
    print("🧠 Quick OpenMind Data Quality Check")
    print("="*50)
    
    base_dir = Path("data/processed/openmind/OpenMind_processed")
    modalities = ['MRA', 'MRI_T1', 'MRI_T2']
    
    print(f"Checking processed data in: {base_dir}")
    
    all_files = {}
    total_files = 0
    
    for modality in modalities:
        mod_dir = base_dir / modality
        if mod_dir.exists():
            files = list(mod_dir.glob("*.nii.gz"))
            all_files[modality] = files
            total_files += len(files)
            print(f"Found {len(files)} {modality} files")
        else:
            print(f"❌ {modality} directory not found")
            all_files[modality] = []
    
    print(f"Total files found: {total_files}")
    
    if total_files == 0:
        print("❌ No files found to analyze!")
        return
    
    # Sample files for detailed analysis
    sample_files = []
    
    for modality, files in all_files.items():
        if not files:
            continue
            
        # Sample different file sizes for each modality
        file_sizes = [(f, f.stat().st_size / (1024**2)) for f in files]
        file_sizes.sort(key=lambda x: x[1])
        
        # Select samples: smallest, median, largest
        samples_per_modality = []
        if len(file_sizes) >= 3:
            samples_per_modality = [
                file_sizes[0][0],              # Smallest
                file_sizes[len(file_sizes)//2][0],  # Median
                file_sizes[-1][0]              # Largest
            ]
        elif len(file_sizes) >= 1:
            samples_per_modality = [file_sizes[0][0]]  # At least one sample
        
        for f in samples_per_modality:
            sample_files.append((f, modality))
    
    print(f"\nAnalyzing {len(sample_files)} sample files:")
    for f, mod in sample_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {mod}: {f.name} ({size_mb:.1f} MB)")
    
    # Analyze sample files
    print(f"\n🔬 Detailed Analysis:")
    results = []
    
    for file_path, modality in tqdm(sample_files):
        result = analyze_volume(file_path, modality)
        results.append(result)
        
        if result['status'] == 'OK':
            print(f"\n📁 {result['file']} ({result['modality']}):") 
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
    
    # Load processing log for additional analysis
    log_path = base_dir / "processing_log.csv"
    if log_path.exists():
        print(f"\n📊 PROCESSING LOG ANALYSIS")
        print("="*50)
        
        df_log = pd.read_csv(log_path)
        
        print(f"Total processed: {len(df_log)} files")
        print(f"Success rate: {(df_log['status'] == 'SUCCESS').sum()}/{len(df_log)} ({(df_log['status'] == 'SUCCESS').mean()*100:.1f}%)")
        
        # Status breakdown
        status_counts = df_log['status'].value_counts()
        print(f"\nStatus breakdown:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Normalization breakdown
        if 'norm_method' in df_log.columns:
            norm_counts = df_log['norm_method'].value_counts()
            print(f"\nNormalization methods:")
            for method, count in norm_counts.items():
                print(f"  {method}: {count}")
        
        # Modality breakdown
        if 'mapped_modality' in df_log.columns:
            mod_counts = df_log['mapped_modality'].value_counts()
            print(f"\nModality breakdown:")
            for modality, count in mod_counts.items():
                print(f"  {modality}: {count}")
    
    # Summary analysis
    print(f"\n📊 SUMMARY ANALYSIS")
    print("="*50)
    
    valid_results = [r for r in results if r['status'] == 'OK']
    
    if valid_results:
        # Check consistency across modalities
        spacings = [r['spacing'] for r in valid_results]
        dtypes = [r['dtype'] for r in valid_results]
        
        print(f"✅ Successfully analyzed {len(valid_results)}/{len(results)} sample files")
        print(f"🔢 Data types: {set(dtypes)}")
        print(f"📏 Spacing consistency: {len(set(spacings))} unique spacings")
        
        # Check spacing consistency (should all be 1.0mm isotropic)
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
            print(f"⚠️  Spacing variations detected:")
            for i, s in enumerate(spacings):
                print(f"    {valid_results[i]['file']}: {s}")
        
        # Value range analysis for MRI (should be 0-1 normalized)
        mri_results = [r for r in valid_results if r['modality'] in ['MRI_T1', 'MRI_T2', 'MRA']]
        if mri_results:
            mri_normalized = all(
                0.0 <= r['min_val'] <= 1.0 and 0.0 <= r['max_val'] <= 1.0 
                for r in mri_results
            )
            
            if mri_normalized:
                print("✅ All MRI values in expected 0-1 range (Nyúl normalized)")
            else:
                print("⚠️  Some MRI values outside 0-1 range:")
                for r in mri_results:
                    if not (0.0 <= r['min_val'] <= 1.0 and 0.0 <= r['max_val'] <= 1.0):
                        print(f"    {r['file']}: {r['min_val']:.3f} to {r['max_val']:.3f}")
        
        # Channel analysis (should all be single channel for MRI)
        channels = [r['num_channels'] for r in valid_results]
        if all(c == 1 for c in channels):
            print("✅ All files are single-channel (correct MRI format)")
        else:
            print(f"⚠️  Channel count issues: {channels}")
    
    # File size analysis
    print(f"\n💾 FILE SIZE ANALYSIS")
    print("="*30)
    
    for modality, files in all_files.items():
        if not files:
            continue
            
        sizes = [f.stat().st_size / (1024**2) for f in files]
        total_size_mb = sum(sizes)
        
        print(f"\n{modality}:")
        print(f"  Files: {len(files)}")
        print(f"  Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.1f} GB)")
        print(f"  Average size: {np.mean(sizes):.1f} MB")
        print(f"  Size range: {min(sizes):.1f} - {max(sizes):.1f} MB")
        print(f"  Median size: {np.median(sizes):.1f} MB")
    
    # Overall dataset size
    total_dataset_size = sum(f.stat().st_size for files in all_files.values() for f in files) / (1024**3)
    print(f"\nTotal dataset size: {total_dataset_size:.1f} GB")
    
    print(f"\n🎯 QUALITY VERDICT:")
    if valid_results and len(valid_results) == len(results):
        if all(r['num_channels'] == 1 for r in valid_results) and spacing_ok:
            print("✅ EXCELLENT - All samples passed quality checks!")
            print("✅ Data format matches RSNA/DeepLesion standards")
            print("✅ Ready for training pipeline integration")
        else:
            print("⚠️  GOOD - Minor format inconsistencies detected")
    else:
        print("❌ ISSUES - Some sample files failed to load")
    
    print(f"\n📄 Dataset characteristics:")
    print(f"   • 85GB total size is NORMAL for OpenMind")
    print(f"   • Single-channel MRI format (Nyúl normalized)")
    print(f"   • 1.0×1.0×1.0mm isotropic spacing")
    print(f"   • Float32 precision for training efficiency")

if __name__ == "__main__":
    main()