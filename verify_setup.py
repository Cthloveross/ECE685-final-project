#!/usr/bin/env python3
"""
Setup verification script for ECE685 Project 2
Run this after installing requirements to verify everything is working.
"""

import sys

def check_python_version():
    """Check Python version"""
    print("=" * 60)
    print("CHECKING PYTHON VERSION")
    print("=" * 60)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print("❌ Python 3.12+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_imports():
    """Check all required imports"""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'sae_lens': 'SAE-Lens',
        'detoxify': 'Detoxify',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'tqdm': 'TQDM',
    }
    
    all_ok = True
    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError:
            print(f"❌ {display_name}: NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("CHECKING GPU SUPPORT")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available (will use CPU)")
            print("   This is OK but slower. Consider using GPU for full dataset.")
    except Exception as e:
        print(f"⚠️  Could not check CUDA: {e}")

def check_directories():
    """Check if required directories exist"""
    print("\n" + "=" * 60)
    print("CHECKING DIRECTORIES")
    print("=" * 60)
    
    import pathlib
    
    dirs = [
        'data/processed',
        'data/results',
        'notebooks',
        'src',
        'docs'
    ]
    
    all_ok = True
    for dir_path in dirs:
        path = pathlib.Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} (missing)")
            all_ok = False
    
    return all_ok

def check_huggingface():
    """Check Hugging Face authentication"""
    print("\n" + "=" * 60)
    print("CHECKING HUGGING FACE AUTHENTICATION")
    print("=" * 60)
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
        return True
    except Exception as e:
        print(f"⚠️  Not logged in to Hugging Face")
        print(f"   Run: huggingface-cli login")
        print(f"   Required for downloading Gemma model")
        return False

def test_basic_functionality():
    """Test basic model loading"""
    print("\n" + "=" * 60)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 60)
    
    try:
        import torch
        print("✅ Creating random tensor...")
        x = torch.randn(2, 3)
        print(f"   Shape: {x.shape}")
        
        from transformers import AutoTokenizer
        print("✅ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        test_text = tokenizer("Hello world")
        print(f"   Tokenized {len(test_text['input_ids'])} tokens")
        
        print("✅ Basic functionality OK")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "🔍 ECE685 PROJECT 2 - SETUP VERIFICATION" + "\n")
    
    results = {
        "Python version": check_python_version(),
        "Dependencies": check_imports(),
        "Directories": check_directories(),
        "Hugging Face": check_huggingface(),
        "Basic functionality": test_basic_functionality(),
    }
    
    # Check CUDA (informational only)
    check_cuda()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to start!")
        print("Run: jupyter notebook")
        print("Then open: notebooks/01_setup_and_data_preparation.ipynb")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above before continuing.")
        print("See docs/QUICK_START.md for troubleshooting.")
    
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
