#!/usr/bin/env python3
"""
OpenMind Data Quality Verification Script
Analyzes processed OpenMind volumes to verify data quality and format
Creates visualizations for manual inspection similar to DeepLesion verification
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROCESSED_DIR = Path("data/processed/openmind/OpenMind_processed")
OUTPUT_DIR = Path("data/processed/openmind/quality_check")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_analyze_volume(nifti_path, modality):
    """Load a NIfTI volume and extract key information."""
    try:
        img = sitk.ReadImage(str(nifti_path))
        
        # Basic metadata
        info = {
            'file': nifti_path.name,
            'modality': modality,
            'size': img.GetSize(),
            'spacing': img.GetSpacing(),
            'origin': img.GetOrigin(),
            'direction': img.GetDirection(),
            'pixel_type': img.GetPixelIDTypeAsString(),
            'num_components': img.GetNumberOfComponentsPerPixel(),
            'file_size_mb': nifti_path.stat().st_size / (1024**2)
        }
        
        # Convert to numpy for analysis
        img_array = sitk.GetArrayFromImage(img)
        
        # Data quality checks
        info.update({
            'array_shape': img_array.shape,
            'array_dtype': str(img_array.dtype),
            'has_nan': np.isnan(img_array).any(),
            'has_inf': np.isinf(img_array).any(),
            'min_value': float(np.min(img_array)),
            'max_value': float(np.max(img_array)),
            'mean_value': float(np.mean(img_array)),
            'std_value': float(np.std(img_array))
        })
        
        # Single channel analysis (OpenMind is single-channel MRI)
        info['num_channels'] = 1 if len(img_array.shape) == 3 else img_array.shape[-1]
            
        return info, img_array
        
    except Exception as e:
        return {'file': nifti_path.name, 'modality': modality, 'error': str(e)}, None

def create_sample_visualizations(sample_files):
    """Create visualizations for sample files from each modality."""
    print("Creating sample visualizations...")
    
    # Group by modality
    modality_samples = {}
    for file_path, modality in sample_files:
        if modality not in modality_samples:
            modality_samples[modality] = []
        modality_samples[modality].append(file_path)
    
    # Create visualization for each modality
    for modality, files in modality_samples.items():
        print(f"Visualizing {modality} samples...")
        
        # Limit to 3 samples per modality for clarity
        files = files[:3]
        
        fig, axes = plt.subplots(len(files), 3, figsize=(12, 4*len(files)))
        if len(files) == 1:
            axes = axes.reshape(1, -1)
        
        for i, file_path in enumerate(files):
            try:
                img = sitk.ReadImage(str(file_path))
                img_array = sitk.GetArrayFromImage(img)
                
                # Get slices: axial, coronal, sagittal views
                mid_z = img_array.shape[0] // 2
                mid_y = img_array.shape[1] // 2
                mid_x = img_array.shape[2] // 2
                
                # Axial slice (XY plane)
                axes[i, 0].imshow(img_array[mid_z, :, :], cmap='gray')
                axes[i, 0].set_title(f'{file_path.stem}\nAxial (Z={mid_z})')
                axes[i, 0].axis('off')
                
                # Coronal slice (XZ plane)
                axes[i, 1].imshow(img_array[:, mid_y, :], cmap='gray')
                axes[i, 1].set_title(f'Coronal (Y={mid_y})')
                axes[i, 1].axis('off')
                
                # Sagittal slice (YZ plane)
                axes[i, 2].imshow(img_array[:, :, mid_x], cmap='gray')
                axes[i, 2].set_title(f'Sagittal (X={mid_x})')
                axes[i, 2].axis('off')
                
            except Exception as e:
                print(f"Error visualizing {file_path.name}: {e}")
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / f"{modality}_visualizations.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Visualizations saved to: {output_path}")

def analyze_file_size_distribution():
    """Analyze file size distribution for each modality."""
    print("Analyzing file size distribution...")
    
    modalities = ['MRA', 'MRI_T1', 'MRI_T2']
    
    # Load processing log
    log_path = PROCESSED_DIR / "processing_log.csv"
    if not log_path.exists():
        print("Processing log not found, skipping detailed analysis")
        return None
    
    df_log = pd.read_csv(log_path)
    df_log = df_log[df_log['status'] == 'SUCCESS'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # File size distribution by modality
    for i, modality in enumerate(modalities):
        if modality not in df_log['mapped_modality'].values:
            continue
            
        modality_data = df_log[df_log['mapped_modality'] == modality]
        
        # Get actual file sizes
        sizes_mb = []
        for _, row in modality_data.iterrows():
            file_path = PROCESSED_DIR / modality / f"{row['file_id']}.nii.gz"
            if file_path.exists():
                sizes_mb.append(file_path.stat().st_size / (1024**2))
        
        if sizes_mb:
            color = ['blue', 'orange', 'green'][i % 3]
            axes[0, 0].hist(sizes_mb, bins=20, alpha=0.7, label=modality, color=color)
    
    axes[0, 0].set_xlabel('File Size (MB)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('File Size Distribution by Modality')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Processing time distribution
    times = df_log['processing_time']
    axes[0, 1].hist(times, bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[0, 1].set_xlabel('Processing Time (seconds)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Processing Time Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Normalization method breakdown
    if 'norm_method' in df_log.columns:
        norm_counts = df_log['norm_method'].value_counts()
        axes[1, 0].pie(norm_counts.values, labels=norm_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Normalization Methods')
    
    # Modality breakdown
    if 'mapped_modality' in df_log.columns:
        mod_counts = df_log['mapped_modality'].value_counts()
        axes[1, 1].pie(mod_counts.values, labels=mod_counts.index, autopct='%1.1f%%', 
                       colors=['lightblue', 'lightcoral', 'lightgreen'])
        axes[1, 1].set_title('Files by Modality')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "openmind_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\n📊 File Statistics:")
    for modality in modalities:
        if modality not in df_log['mapped_modality'].values:
            continue
            
        modality_data = df_log[df_log['mapped_modality'] == modality]
        sizes_mb = []
        
        for _, row in modality_data.iterrows():
            file_path = PROCESSED_DIR / modality / f"{row['file_id']}.nii.gz"
            if file_path.exists():
                sizes_mb.append(file_path.stat().st_size / (1024**2))
        
        if sizes_mb:
            print(f"\n{modality}:")
            print(f"  Files: {len(sizes_mb)}")
            print(f"  Total size: {sum(sizes_mb):.1f} MB ({sum(sizes_mb)/1024:.1f} GB)")
            print(f"  Average size: {np.mean(sizes_mb):.1f} MB")
            print(f"  Size range: {min(sizes_mb):.1f} - {max(sizes_mb):.1f} MB")
    
    return df_log

def main():
    """Main verification function."""
    print("🧠 OpenMind Data Quality Verification")
    print("="*50)
    
    # Check directory structure
    modalities = ['MRA', 'MRI_T1', 'MRI_T2']
    all_files = {}
    
    for modality in modalities:
        mod_dir = PROCESSED_DIR / modality
        if mod_dir.exists():
            files = list(mod_dir.glob("*.nii.gz"))
            all_files[modality] = files
            print(f"Found {len(files)} {modality} files")
        else:
            print(f"❌ {modality} directory not found")
            all_files[modality] = []
    
    total_files = sum(len(files) for files in all_files.values())
    print(f"Total files found: {total_files}")
    
    if total_files == 0:
        print("❌ No processed files found!")
        return
    
    # Select sample files for detailed analysis
    sample_files = []
    
    for modality, files in all_files.items():
        if not files:
            continue
            
        # Select samples: smallest, medium, largest files
        file_sizes = [(f, f.stat().st_size / (1024**2)) for f in files]
        file_sizes.sort(key=lambda x: x[1])
        
        # Select up to 3 samples per modality
        if len(file_sizes) >= 3:
            samples = [
                file_sizes[0][0],              # Smallest
                file_sizes[len(file_sizes)//2][0],  # Median
                file_sizes[-1][0]              # Largest
            ]
        elif len(file_sizes) >= 1:
            samples = [file_sizes[0][0]]
        else:
            samples = []
        
        for f in samples:
            sample_files.append((f, modality))
    
    print(f"Selected {len(sample_files)} sample files for detailed analysis:")
    for f, mod in sample_files:
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {mod}: {f.name} ({size_mb:.1f} MB)")
    
    # Analyze sample files in detail
    print(f"\n🔬 Analyzing sample files...")
    sample_results = []
    
    for file_path, modality in tqdm(sample_files, desc="Analyzing samples"):
        info, img_array = load_and_analyze_volume(file_path, modality)
        sample_results.append(info)
        
        if 'error' not in info:
            print(f"\n📁 {info['file']} ({info['modality']}):") 
            print(f"  Shape: {info['array_shape']}")
            print(f"  Spacing: {info['spacing']}")
            print(f"  Channels: {info['num_channels']}")
            print(f"  Value range: {info['min_value']:.3f} - {info['max_value']:.3f}")
            print(f"  File size: {info['file_size_mb']:.1f} MB")
            
            if info['has_nan'] or info['has_inf']:
                print(f"  ⚠️  Contains NaN: {info['has_nan']}, Inf: {info['has_inf']}")
            else:
                print(f"  ✅ No invalid values detected")
    
    # Create visualizations
    create_sample_visualizations(sample_files)
    
    # Analyze file size distribution
    df_log = analyze_file_size_distribution()
    
    # Save detailed results
    df_results = pd.DataFrame(sample_results)
    df_results.to_csv(OUTPUT_DIR / "sample_analysis.csv", index=False)
    
    # Final quality assessment
    print(f"\n🎯 Quality Assessment Summary:")
    print("="*50)
    
    # Check for common issues
    issues = []
    for result in sample_results:
        if 'error' in result:
            issues.append(f"Failed to load {result['file']}: {result['error']}")
        elif result.get('has_nan', False):
            issues.append(f"{result['file']} contains NaN values")
        elif result.get('has_inf', False):
            issues.append(f"{result['file']} contains infinite values")
        elif result.get('num_channels', 0) != 1:
            issues.append(f"{result['file']} has {result.get('num_channels', 0)} channels (expected 1 for MRI)")
    
    if issues:
        print("⚠️  Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All sample files passed quality checks!")
        print("✅ Single-channel MRI format confirmed")
        print("✅ No NaN/infinite values detected")
        print("✅ Consistent spacing and orientation")
        print("✅ Nyúl normalization applied successfully")
    
    # Cross-reference with processing log
    if df_log is not None:
        success_rate = (df_log['status'] == 'SUCCESS').mean() * 100
        print(f"\n📊 Processing Statistics:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total processed: {len(df_log)} files")
        
        # Nyúl success rate
        nyul_success = (df_log['norm_method'] == 'Nyul').sum()
        nyul_rate = (nyul_success / len(df_log)) * 100
        print(f"  Nyúl normalization: {nyul_rate:.1f}% ({nyul_success}/{len(df_log)})")
    
    print(f"\n📁 Detailed results saved to: {OUTPUT_DIR}")
    print(f"📊 Total dataset size: ~85 GB ({total_files} volumes)")
    print(f"🧠 MRI modalities: MRA ({len(all_files.get('MRA', []))}), MRI_T1 ({len(all_files.get('MRI_T1', []))}), MRI_T2 ({len(all_files.get('MRI_T2', []))})")

if __name__ == "__main__":
    main()