"""
Test script to verify installation and basic functionality
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'torch_geometric': 'PyTorch Geometric',
        'timm': 'timm',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
    }
    
    optional_packages = {
        'wandb': 'Weights & Biases',
        'tensorboard': 'TensorBoard',
    }
    
    failed = []
    
    # Test required packages
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - REQUIRED")
            failed.append(name)
    
    # Test optional packages
    for package, name in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - Optional (recommended)")
    
    if failed:
        print(f"\n❌ Failed to import required packages: {', '.join(failed)}")
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages imported successfully!")
        return True


def test_modules():
    """Test if all project modules can be imported"""
    print("\nTesting project modules...")
    
    modules = [
        'config',
        'utils',
        'augmentation',
        'dataset',
        'knn_builder',
        'gpcn_layer',
        'model',
        'losses',
        'metrics',
        'visualization',
        'trainer',
    ]
    
    failed = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name}.py")
        except ImportError as e:
            print(f"  ✗ {module_name}.py - {str(e)}")
            failed.append(module_name)
    
    if failed:
        print(f"\n❌ Failed to import modules: {', '.join(failed)}")
        return False
    else:
        print("\n✅ All project modules imported successfully!")
        return True


def test_config():
    """Test configuration creation"""
    print("\nTesting configuration...")
    
    try:
        from config import get_default_config, get_quick_test_config, get_full_config
        
        config = get_default_config()
        print("  ✓ Default config created")
        
        config = get_quick_test_config()
        print("  ✓ Quick test config created")
        
        config = get_full_config()
        print("  ✓ Full config created")
        
        print("\n✅ Configuration system working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        return False


def test_model():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        import torch
        from config import get_default_config
        from model import create_model
        
        config = get_default_config()
        config.model.use_uncertainty = True  # Simpler test
        config.model.num_gpcn_layers = 3
        config.model.use_multi_scale = True
        
        print("  Creating model...")
        model = create_model(config)
        print("  ✓ Model created")
        
        # Test forward pass
        print("  Testing forward pass...")
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {x.shape}")
        print(f"    Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        
        print("\n✅ Model creation and forward pass working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device count: {torch.cuda.device_count()}")
            print(f"    Device name: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
            
            # Test simple operation on GPU
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            print(f"  ✓ GPU computation working")
        else:
            print(f"  ⚠ CUDA not available - will use CPU")
            print(f"    This is fine but training will be slower")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CUDA test failed: {e}")
        return False


def test_augmentation():
    """Test data augmentation"""
    print("\nTesting data augmentation...")
    
    try:
        from PIL import Image
        import numpy as np
        from augmentation import HistologyAugmentation, ValidationTransform
        
        # Create dummy image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Test training augmentation
        train_aug = HistologyAugmentation(
            use_stain_aug=False,  # Disable for quick test
            use_elastic=False,
            use_gridmask=False
        )
        
        img_aug = train_aug(img)
        print(f"  ✓ Training augmentation working")
        print(f"    Output shape: {img_aug.shape}")
        
        # Test validation transform
        val_transform = ValidationTransform()
        img_val = val_transform(img)
        print(f"  ✓ Validation transform working")
        print(f"    Output shape: {img_val.shape}")
        
        print("\n✅ Data augmentation working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("GPCN-ViT Installation and Functionality Test")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Package imports", test_imports()))
    results.append(("Project modules", test_modules()))
    results.append(("Configuration", test_config()))
    results.append(("CUDA support", test_cuda()))
    results.append(("Data augmentation", test_augmentation()))
    results.append(("Model creation", test_model()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Prepare your dataset")
        print("  2. Update data path in config.py or use --data-root argument")
        print("  3. Run training: python train.py --experiment-name test_run")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
