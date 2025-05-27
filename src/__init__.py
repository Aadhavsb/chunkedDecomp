"""ChunkedDecomp: Efficient KV cache compression for transformers."""

__version__ = "0.1.0"

# Import main classes for easy access
from .models import ChunkedDecomp, ChunkedKVCache, CompressedModelWrapper
from .utils import SVDCompressor, MemoryTracker, ComprehensiveBenchmark, DatasetManager
from .evaluation import PerformanceEvaluator, MemoryProfiler

__all__ = [
    # Main classes
    'ChunkedDecomp',
    'ChunkedKVCache', 
    'CompressedModelWrapper',
    # Utilities
    'SVDCompressor',
    'MemoryTracker',
    'ComprehensiveBenchmark',
    'DatasetManager',
    # Evaluation
    'PerformanceEvaluator',
    'MemoryProfiler'
]
