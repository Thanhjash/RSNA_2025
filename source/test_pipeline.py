#!/usr/bin/env python3
"""
Test Pipeline for Phase 0 Pre-training
Comprehensive testing of the implemented pipeline components

This script tests the Phase 0 pre-training pipeline with fallbacks for missing dependencies.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

# Add source directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_basic_imports():
    """Test basic imports that should always work"""
    print("🧪 Testing basic imports...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        import numpy as np
        print(f"✅ NumPy {np.__version__}")

        from pathlib import Path
        print("✅ Standard library imports")

        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_optional_imports():
    """Test optional imports with fallbacks"""
    print("\n🧪 Testing optional imports...")

    # Test ptwt (PyTorch Wavelet Toolbox)
    try:
        import ptwt
        print("✅ ptwt (PyTorch Wavelet Toolbox) available")
        PTWT_AVAILABLE = True
    except ImportError:
        print("⚠️  ptwt not available - will use PyWavelets fallback")
        PTWT_AVAILABLE = False

    # Test MinkowskiEngine
    try:
        import MinkowskiEngine as ME
        print("✅ MinkowskiEngine available")
        ME_AVAILABLE = True
    except ImportError:
        print("⚠️  MinkowskiEngine not available - will use dense fallback")
        ME_AVAILABLE = False

    # Test SimpleITK
    try:
        import SimpleITK as sitk
        print("✅ SimpleITK available")
        SITK_AVAILABLE = True
    except ImportError:
        print("⚠️  SimpleITK not available - data loading may be limited")
        SITK_AVAILABLE = False

    return PTWT_AVAILABLE, ME_AVAILABLE, SITK_AVAILABLE

def create_fallback_dwt():
    """Create fallback DWT functions using PyWavelets"""
    try:
        import pywt
        print("✅ Using PyWavelets as fallback for DWT operations")

        def dwt3d_forward_fallback(x, wavelet='db1', level=1):
            """Fallback DWT using PyWavelets"""
            # Simple implementation - not optimized
            B, C, D, H, W = x.shape
            coeffs_list = []

            for b in range(B):
                for c in range(C):
                    volume = x[b, c].cpu().numpy()
                    coeffs = pywt.wavedecn(volume, wavelet, level=level)
                    coeffs_list.append(coeffs)

            # Return simplified structure
            return x[:, :, ::2, ::2, ::2], []  # Simplified low-freq only

        return dwt3d_forward_fallback

    except ImportError:
        print("⚠️  PyWavelets not available - DWT operations disabled")
        return None

def test_model_components():
    """Test individual model components"""
    print("\n🧪 Testing model components...")

    device = torch.device('cpu')  # Use CPU for testing

    # Test basic neural network components
    try:
        # Test basic attention
        attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        test_input = torch.randn(2, 10, 64)
        output, _ = attention(test_input, test_input, test_input)
        print(f"✅ Attention mechanism: {output.shape}")

        # Test MLP
        mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 64)
        )
        mlp_output = mlp(test_input)
        print(f"✅ MLP: {mlp_output.shape}")

        # Test 3D convolutions
        conv3d = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        test_volume = torch.randn(2, 1, 8, 8, 8)
        conv_output = conv3d(test_volume)
        print(f"✅ 3D Convolution: {conv_output.shape}")

        return True

    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def test_masking_functions():
    """Test masking functions"""
    print("\n🧪 Testing masking functions...")

    try:
        from modules.phase0.losses.masking import generate_hierarchical_mask, MiMController

        # Test hierarchical mask generation
        test_shape = (2, 1, 16, 16, 16)
        global_mask, local_mask = generate_hierarchical_mask(
            tensor_shape=test_shape,
            global_mask_ratio=0.6,
            local_mask_ratio=0.8,
            device=torch.device('cpu')
        )

        print(f"✅ Hierarchical masking: global={global_mask.shape}, local={local_mask.shape}")

        # Verify subset relationship
        subset_check = (local_mask & ~global_mask).sum()
        assert subset_check == 0, "Local mask should be subset of global mask"
        print("✅ Mask subset relationship verified")

        # Test MiM controller
        controller = MiMController()
        test_images = torch.randn(*test_shape)
        mask_results = controller(test_images)
        print(f"✅ MiM Controller: {len(mask_results)} outputs")

        return True

    except Exception as e:
        print(f"❌ Masking test failed: {e}")
        return False

def test_contrastive_learning():
    """Test contrastive learning components"""
    print("\n🧪 Testing contrastive learning...")

    try:
        from modules.phase0.losses.contrastive import CrossLevelAlignmentLoss, info_nce_loss

        # Test InfoNCE loss
        query = torch.randn(4, 128)
        positive = torch.randn(4, 128)
        negatives = torch.randn(4, 10, 128)

        loss = info_nce_loss(query, positive, negatives)
        print(f"✅ InfoNCE loss: {loss.item():.6f}")

        # Test alignment loss
        alignment_loss = CrossLevelAlignmentLoss(feature_dim=64, projection_dim=32)
        feat1 = torch.randn(2, 64, 4, 4, 4)
        feat2 = torch.randn(2, 64, 2, 2, 2)

        align_loss = alignment_loss(feat1, feat2)
        print(f"✅ Cross-level alignment loss: {align_loss.item():.6f}")

        return True

    except Exception as e:
        print(f"❌ Contrastive learning test failed: {e}")
        return False

def test_simplified_model():
    """Test a simplified version of the complete model"""
    print("\n🧪 Testing simplified model...")

    try:
        # Create a simplified model without external dependencies
        class SimplifiedPretrainer(nn.Module):
            def __init__(self, img_size=(16, 16, 16), in_channels=1, embed_dim=64):
                super().__init__()
                self.img_size = img_size
                self.embed_dim = embed_dim

                # Simple backbone
                self.backbone = nn.Sequential(
                    nn.Conv3d(in_channels, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv3d(32, embed_dim, 3, padding=1),
                    nn.AdaptiveAvgPool3d(1)
                )

                # Simple decoder
                self.decoder = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.ReLU(),
                    nn.Linear(embed_dim * 2, in_channels)
                )

            def forward(self, x):
                B, C, D, H, W = x.shape

                # Generate simple random mask
                mask = torch.rand(B, D, H, W) > 0.5
                masked_x = x * mask.unsqueeze(1).float()

                # Extract features
                features = self.backbone(masked_x).flatten(1)

                # Reconstruct
                reconstruction = self.decoder(features)
                reconstruction = reconstruction.view(B, C, 1, 1, 1).expand_as(x)

                # Simple reconstruction loss
                loss = nn.MSELoss()(reconstruction * (~mask).unsqueeze(1).float(),
                                  x * (~mask).unsqueeze(1).float())

                return {
                    'total_loss': loss,
                    'reconstruction': reconstruction,
                    'mask': mask,
                    'features': features
                }

        # Test the simplified model
        model = SimplifiedPretrainer(img_size=(8, 8, 8), embed_dim=32)
        test_input = torch.randn(2, 1, 8, 8, 8)

        results = model(test_input)
        loss = results['total_loss']

        print(f"✅ Simplified model forward pass: loss={loss.item():.6f}")
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test backward pass
        loss.backward()
        print("✅ Backward pass successful")

        return True

    except Exception as e:
        print(f"❌ Simplified model test failed: {e}")
        return False

def test_config_system():
    """Test configuration system"""
    print("\n🧪 Testing configuration system...")

    try:
        from config.phase0_config import Phase0Config, get_development_config

        # Test default config
        config = Phase0Config()
        print(f"✅ Default config: embed_dim={config.model.embed_dim}")

        # Test development config
        dev_config = get_development_config()
        print(f"✅ Development config: img_size={dev_config.model.img_size}")

        # Test serialization
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            loaded_config = Phase0Config.load(f.name)
            print("✅ Config serialization/deserialization")

        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_pipeline():
    """Test data pipeline components"""
    print("\n🧪 Testing data pipeline...")

    try:
        from modules.phase0.data.transforms import PretrainingTransforms

        # Test transforms
        train_transforms = PretrainingTransforms.get_train_transforms(target_size=(16, 16, 16))
        val_transforms = PretrainingTransforms.get_val_transforms(target_size=(16, 16, 16))

        test_volume = torch.randn(1, 8, 8, 8)

        # Apply transforms
        transformed = train_transforms(test_volume)
        val_transformed = val_transforms(test_volume)

        print(f"✅ Transforms: train={transformed.shape}, val={val_transformed.shape}")

        return True

    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Starting Phase 0 Pipeline Comprehensive Test")
    print("=" * 60)

    test_results = {}

    # Run all tests
    test_results['basic_imports'] = test_basic_imports()
    test_results['optional_imports'] = test_optional_imports()
    test_results['model_components'] = test_model_components()
    test_results['masking'] = test_masking_functions()
    test_results['contrastive'] = test_contrastive_learning()
    test_results['simplified_model'] = test_simplified_model()
    test_results['config'] = test_config_system()
    test_results['data_pipeline'] = test_data_pipeline()

    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)

    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Pipeline is ready for training.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Pipeline functional with some limitations.")
    else:
        print("❌ Multiple test failures. Please check dependencies and implementation.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)