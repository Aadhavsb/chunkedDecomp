#!/usr/bin/env python3
"""
Quick start script for ChunkedDecomp
Run this to verify your installation and see basic functionality
"""

import os
import sys
import traceback
from typing import Dict, Any

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('pandas', 'Pandas'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies found!")
    return True

def check_pytorch_setup():
    """Check PyTorch and CUDA setup."""
    print("\n🔍 Checking PyTorch setup...")
    
    try:
        import torch
        print(f"  ✅ PyTorch version: {torch.__version__}")
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA device count: {torch.cuda.device_count()}")
            print(f"  ✅ Current device: {torch.cuda.current_device()}")
        else:
            print("  ⚠️  CUDA not available - will use CPU (slower)")
        
        return True
    except Exception as e:
        print(f"  ❌ PyTorch setup error: {e}")
        return False

def test_basic_import():
    """Test basic package imports."""
    print("\n🔍 Testing package imports...")
    
    try:
        # Add src to path if needed
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from src import ChunkedDecomp, MemoryTracker, PerformanceEvaluator
        print("  ✅ ChunkedDecomp imported successfully")
        
        from src.utils import SVDCompressor
        print("  ✅ SVDCompressor imported successfully")
        
        from src.models import ChunkedKVCache
        print("  ✅ ChunkedKVCache imported successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic compression functionality."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from src import ChunkedDecomp, MemoryTracker
        
        # Use smallest possible model for testing
        model_name = "distilgpt2"
        print(f"  📥 Loading model: {model_name}")
        
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("  ✅ Model loaded successfully")
        
        # Test compression
        print("  🗜️  Testing compression...")
        compressor = ChunkedDecomp(
            compression_ratio=0.7,  # Conservative compression for testing
            chunk_size=32,
            adaptive_rank=True
        )
        
        with MemoryTracker() as tracker:
            compressed_model = compressor.compress_model(model)
        
        print("  ✅ Compression successful")
        
        # Test inference
        print("  🧠 Testing inference...")
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        
        # Original model inference
        with torch.no_grad():
            original_outputs = model(**inputs)
        
        # Compressed model inference  
        with torch.no_grad():
            compressed_outputs = compressed_model(**inputs)
        
        print("  ✅ Inference successful")
        
        # Calculate metrics
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        actual_ratio = compressed_params / original_params
        
        print(f"  📊 Original parameters: {original_params:,}")
        print(f"  📊 Compressed parameters: {compressed_params:,}")
        print(f"  📊 Actual compression ratio: {actual_ratio:.3f}")
        print(f"  📊 Memory reduction: {(1-actual_ratio)*100:.1f}%")
        print(f"  📊 Peak memory usage: {tracker.get_peak_memory():.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_advanced_features():
    """Test advanced features."""
    print("\n🔍 Testing advanced features...")
    
    try:
        from src.evaluation import PerformanceEvaluator, MemoryProfiler
        from src.utils import ComprehensiveBenchmark, DatasetManager
        
        print("  ✅ Performance evaluation tools imported")
        
        # Test memory profiler
        profiler = MemoryProfiler()
        print("  ✅ Memory profiler initialized")
        
        # Test benchmark utilities
        benchmark = ComprehensiveBenchmark()
        print("  ✅ Benchmark utilities initialized")
        
        # Test dataset manager
        data_manager = DatasetManager()
        print("  ✅ Dataset manager initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Advanced features test failed: {e}")
        return False

def run_sample_compression():
    """Run a sample compression workflow."""
    print("\n🚀 Running sample compression workflow...")
    
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from src import ChunkedDecomp, MemoryTracker
        from src.evaluation import PerformanceEvaluator
        
        # Load model
        model_name = "distilgpt2"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Multiple compression ratios
        compression_ratios = [0.8, 0.6, 0.4]
        results = []
        
        for ratio in compression_ratios:
            print(f"  🗜️  Testing compression ratio: {ratio}")
            
            compressor = ChunkedDecomp(
                compression_ratio=ratio,
                chunk_size=64,
                adaptive_rank=True
            )
            
            with MemoryTracker() as tracker:
                compressed_model = compressor.compress_model(model)
                
                # Test inference
                test_text = "The future of AI is"
                inputs = tokenizer(test_text, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = compressed_model(**inputs)
            
            # Calculate metrics
            original_params = sum(p.numel() for p in model.parameters())
            compressed_params = sum(p.numel() for p in compressed_model.parameters())
            actual_ratio = compressed_params / original_params
            
            result = {
                'target_ratio': ratio,
                'actual_ratio': actual_ratio,
                'memory_reduction': (1 - actual_ratio) * 100,
                'peak_memory': tracker.get_peak_memory()
            }
            results.append(result)
            
            print(f"    📊 Actual ratio: {actual_ratio:.3f}")
            print(f"    📊 Memory reduction: {result['memory_reduction']:.1f}%")
        
        # Summary
        print("\n📊 Compression Summary:")
        print("Target → Actual → Memory Reduction")
        for r in results:
            print(f"  {r['target_ratio']:.1f} → {r['actual_ratio']:.3f} → {r['memory_reduction']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sample workflow failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main function to run all tests."""
    print("🚀 ChunkedDecomp Quick Start Test\n")
    print("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("PyTorch Setup", check_pytorch_setup), 
        ("Package Imports", test_basic_import),
        ("Basic Functionality", test_basic_functionality),
        ("Advanced Features", test_advanced_features),
        ("Sample Workflow", run_sample_compression)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ChunkedDecomp is ready to use!")
        print("\nNext steps:")
        print("1. Try the Jupyter notebook: jupyter notebook notebooks/chunked_decomp_exploration.ipynb")
        print("2. Run a compression script: python scripts/run_compression.py --model_name distilgpt2")
        print("3. Check the README for detailed usage instructions")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Try installing missing dependencies or check your Python environment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
