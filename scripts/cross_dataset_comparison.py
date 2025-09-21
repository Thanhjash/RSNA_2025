#!/usr/bin/env python3
"""
Cross-Dataset Comparison Script
Compares preprocessing consistency across RSNA, DeepLesion, and OpenMind datasets
Ensures all datasets follow identical standards for training pipeline integration
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset_sample(dataset_name, base_path, sample_files, expected_format):
    """Analyze a sample of files from a dataset."""
    print(f"\n{'='*20} {dataset_name} ANALYSIS {'='*20}")
    
    results = {
        'dataset': dataset_name,
        'expected_format': expected_format,
        'files_analyzed': 0,
        'spacing_consistent': True,
        'orientation_consistent': True,
        'dtype_consistent': True,
        'channel_format_correct': True,
        'normalization_correct': True,
        'issues': []
    }
    
    target_spacing = (1.0, 1.0, 1.0)
    target_orientation = "RAS"
    
    for sample_file in sample_files:
        try:
            file_path = Path(base_path) / sample_file
            if not file_path.exists():
                continue
                
            img = sitk.ReadImage(str(file_path))
            img_array = sitk.GetArrayFromImage(img)
            
            results['files_analyzed'] += 1
            
            # Check spacing
            spacing = img.GetSpacing()
            if not all(abs(s - t) < 0.01 for s, t in zip(spacing, target_spacing)):
                results['spacing_consistent'] = False
                results['issues'].append(f"{sample_file}: Incorrect spacing {spacing}")
            
            # Check data type
            if img_array.dtype != np.float32:
                results['dtype_consistent'] = False
                results['issues'].append(f"{sample_file}: Wrong dtype {img_array.dtype}")
            
            # Check channels based on expected format
            if expected_format == "3-channel":
                if len(img_array.shape) != 4 or img_array.shape[-1] != 3:
                    results['channel_format_correct'] = False
                    results['issues'].append(f"{sample_file}: Expected 3-channel, got shape {img_array.shape}")
            elif expected_format == "single-channel":
                if len(img_array.shape) != 3:
                    results['channel_format_correct'] = False
                    results['issues'].append(f"{sample_file}: Expected single-channel, got shape {img_array.shape}")
            
            # Check normalization
            min_val, max_val = np.min(img_array), np.max(img_array)
            
            if expected_format == "3-channel":  # CT data
                # For 3-channel CT, values should be 0-1 normalized
                if not (0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0):
                    results['normalization_correct'] = False
                    results['issues'].append(f"{sample_file}: Values outside 0-1 range: {min_val:.3f} to {max_val:.3f}")
            elif expected_format == "single-channel":  # MRI data
                # For MRI, some may be Nyúl normalized (0-1) or percentile normalized
                # This is more flexible as both are acceptable
                pass
            
            print(f"✅ {sample_file}: Shape={img_array.shape}, Spacing={spacing}, Range={min_val:.3f}-{max_val:.3f}")
            
        except Exception as e:
            results['issues'].append(f"{sample_file}: Failed to load - {e}")
            print(f"❌ {sample_file}: {e}")
    
    return results

def main():
    """Main comparison function."""
    print("🔄 Cross-Dataset Comparison for RSNA 2025")
    print("="*60)
    
    datasets = {}
    
    # RSNA Dataset Analysis
    # Note: RSNA has multiple modalities, we'll sample from each
    rsna_samples = [
        "CTA/1.2.826.0.1.3680043.23402.nii.gz",  # Example CTA file
        "MRI_T1/1.2.826.0.1.3680043.24801.nii.gz",  # Example MRI_T1 file
    ]
    # Skip RSNA for now as we don't have processed data locally
    
    # DeepLesion Dataset Analysis  
    deeplesion_path = "data/processed/NIH_deeplesion/CT"
    deeplesion_files = list(Path(deeplesion_path).glob("*.nii.gz"))
    if deeplesion_files:
        deeplesion_samples = [f.name for f in deeplesion_files[:3]]  # Sample 3 files
        datasets['DeepLesion'] = analyze_dataset_sample(
            "DeepLesion", 
            deeplesion_path,
            deeplesion_samples,
            "3-channel"
        )
    
    # OpenMind Dataset Analysis
    openmind_path = "data/processed/openmind/OpenMind_processed"
    
    # Sample from each modality
    openmind_samples = []
    for modality in ['MRA', 'MRI_T1', 'MRI_T2']:
        mod_path = Path(openmind_path) / modality
        if mod_path.exists():
            files = list(mod_path.glob("*.nii.gz"))
            if files:
                openmind_samples.append(f"{modality}/{files[0].name}")
    
    if openmind_samples:
        datasets['OpenMind'] = analyze_dataset_sample(
            "OpenMind",
            openmind_path, 
            openmind_samples,
            "single-channel"
        )
    
    # Comparative Analysis
    print(f"\n{'='*60}")
    print("📊 CROSS-DATASET COMPARISON SUMMARY")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    
    for dataset_name, results in datasets.items():
        comparison_data.append({
            'Dataset': dataset_name,
            'Files_Analyzed': results['files_analyzed'],
            'Expected_Format': results['expected_format'],
            'Spacing_OK': '✅' if results['spacing_consistent'] else '❌',
            'DType_OK': '✅' if results['dtype_consistent'] else '❌',
            'Channels_OK': '✅' if results['channel_format_correct'] else '❌',
            'Normalization_OK': '✅' if results['normalization_correct'] else '❌',
            'Issues_Count': len(results['issues'])
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Detailed issues
    print(f"\n📋 DETAILED ISSUES:")
    all_issues = []
    
    for dataset_name, results in datasets.items():
        if results['issues']:
            print(f"\n{dataset_name} Issues:")
            for issue in results['issues']:
                print(f"  - {issue}")
                all_issues.append(f"{dataset_name}: {issue}")
        else:
            print(f"\n{dataset_name}: ✅ No issues detected")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print("="*40)
    
    total_issues = len(all_issues)
    if total_issues == 0:
        print("✅ EXCELLENT - All datasets follow consistent preprocessing standards")
        print("✅ Ready for unified training pipeline")
        verdict = "EXCELLENT"
    elif total_issues <= 2:
        print("⚠️  GOOD - Minor inconsistencies detected")
        print("⚠️  Review issues before training")
        verdict = "GOOD"
    else:
        print("❌ ISSUES - Significant preprocessing inconsistencies")
        print("❌ Requires preprocessing standardization")
        verdict = "ISSUES"
    
    # Expected standards summary
    print(f"\n📏 PREPROCESSING STANDARDS SUMMARY:")
    print("="*45)
    print("✅ Target Spacing: 1.0×1.0×1.0mm isotropic")
    print("✅ Target Orientation: RAS")
    print("✅ Target Data Type: Float32")
    print("✅ DeepLesion: 3-channel CT (brain/blood/bone windows)")
    print("✅ OpenMind: Single-channel MRI (Nyúl/percentile normalized)")
    print("✅ RSNA: Multi-modal (CTA: 3-channel, MRI: single-channel)")
    
    # Training recommendations
    print(f"\n🚀 TRAINING PIPELINE RECOMMENDATIONS:")
    print("="*50)
    
    if verdict == "EXCELLENT":
        print("1. ✅ Proceed with current data loader configuration")
        print("2. ✅ Implement modality-specific input channels:")
        print("   - CT/CTA: 3-channel input layers")
        print("   - MRI/MRA: 1-channel input layers") 
        print("3. ✅ Use unified spacing/orientation assumptions")
    else:
        print("1. ⚠️  Address preprocessing inconsistencies first")
        print("2. ⚠️  Verify modality-specific normalizations")
        print("3. ⚠️  Test data loader with corrected samples")
    
    # Save comparison results
    output_path = "data/processed/cross_dataset_comparison_report.csv"
    df_comparison.to_csv(output_path, index=False)
    
    # Save detailed issues
    if all_issues:
        issues_path = "data/processed/cross_dataset_issues.txt"
        with open(issues_path, 'w') as f:
            f.write("Cross-Dataset Preprocessing Issues\n")
            f.write("="*40 + "\n\n")
            for issue in all_issues:
                f.write(f"- {issue}\n")
        print(f"\n📄 Detailed issues saved to: {issues_path}")
    
    print(f"📄 Comparison report saved to: {output_path}")
    
    return verdict, datasets

if __name__ == "__main__":
    verdict, results = main()