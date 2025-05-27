"""Memory utilities for tracking and analyzing memory usage."""

import torch
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Tracks memory usage during model execution."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory tracker.
        
        Args:
            device: Device to track. If None, tracks all available devices.
        """
        self.device = device
        self.snapshots = []
        self.start_memory = None
        
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        info = {
            'timestamp': time.time(),
            'cpu_memory': self._get_cpu_memory(),
        }
        
        if torch.cuda.is_available():
            if self.device is not None and self.device.type == 'cuda':
                info['gpu_memory'] = self._get_gpu_memory(self.device.index)
            else:
                # Get memory for all GPUs
                info['gpu_memory'] = {}
                for i in range(torch.cuda.device_count()):
                    info['gpu_memory'][f'cuda:{i}'] = self._get_gpu_memory(i)
        
        return info
    
    def _get_cpu_memory(self) -> Dict[str, float]:
        """Get CPU memory information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'available_mb': virtual_memory.available / 1024 / 1024,
            'percent_used': virtual_memory.percent
        }
    
    def _get_gpu_memory(self, device_index: int = 0) -> Dict[str, float]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {}
        
        with torch.cuda.device(device_index):
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            
            # Get total memory
            total_memory = torch.cuda.get_device_properties(device_index).total_memory
            
            return {
                'allocated_mb': allocated / 1024 / 1024,
                'reserved_mb': reserved / 1024 / 1024,
                'max_allocated_mb': max_allocated / 1024 / 1024,
                'max_reserved_mb': max_reserved / 1024 / 1024,
                'total_mb': total_memory / 1024 / 1024,
                'free_mb': (total_memory - reserved) / 1024 / 1024,
                'utilization_percent': (allocated / total_memory) * 100
            }
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        memory_info = self.get_memory_info()
        memory_info['label'] = label
        self.snapshots.append(memory_info)
        
        logger.debug(f"Memory snapshot '{label}': {memory_info}")
        return memory_info
    
    def start_tracking(self):
        """Start memory tracking."""
        self.start_memory = self.snapshot("start")
        return self.start_memory
    
    def stop_tracking(self):
        """Stop tracking and return memory difference."""
        if self.start_memory is None:
            logger.warning("Memory tracking was not started")
            return None
        
        end_memory = self.snapshot("end")
        
        # Calculate differences
        diff = self._calculate_memory_diff(self.start_memory, end_memory)
        
        return {
            'start': self.start_memory,
            'end': end_memory,
            'difference': diff,
            'snapshots': self.snapshots
        }
    
    def _calculate_memory_diff(self, start: Dict, end: Dict) -> Dict[str, Any]:
        """Calculate memory usage difference."""
        diff = {}
        
        # CPU memory difference
        if 'cpu_memory' in start and 'cpu_memory' in end:
            cpu_start = start['cpu_memory']
            cpu_end = end['cpu_memory']
            diff['cpu_memory'] = {
                'rss_mb_diff': cpu_end['rss_mb'] - cpu_start['rss_mb'],
                'vms_mb_diff': cpu_end['vms_mb'] - cpu_start['vms_mb']
            }
        
        # GPU memory difference
        if 'gpu_memory' in start and 'gpu_memory' in end:
            gpu_start = start['gpu_memory']
            gpu_end = end['gpu_memory']
            
            if isinstance(gpu_start, dict) and isinstance(gpu_end, dict):
                diff['gpu_memory'] = {}
                for device in gpu_start:
                    if device in gpu_end:
                        diff['gpu_memory'][device] = {
                            'allocated_mb_diff': gpu_end[device]['allocated_mb'] - gpu_start[device]['allocated_mb'],
                            'reserved_mb_diff': gpu_end[device]['reserved_mb'] - gpu_start[device]['reserved_mb']
                        }
            else:
                # Single GPU case
                diff['gpu_memory'] = {
                    'allocated_mb_diff': gpu_end['allocated_mb'] - gpu_start['allocated_mb'],
                    'reserved_mb_diff': gpu_end['reserved_mb'] - gpu_start['reserved_mb']
                }
        
        return diff
    
    def clear_snapshots(self):
        """Clear all snapshots."""
        self.snapshots.clear()
        self.start_memory = None


@contextmanager
def memory_profiler(device: Optional[torch.device] = None, label: str = "operation"):
    """Context manager for memory profiling.
    
    Args:
        device: Device to profile
        label: Label for the operation
        
    Usage:
        with memory_profiler() as tracker:
            # Your code here
            pass
        print(tracker.result)
    """
    tracker = MemoryTracker(device)
    tracker.start_tracking()
    
    try:
        yield tracker
    finally:
        tracker.result = tracker.stop_tracking()


def force_garbage_collection():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MemoryBenchmark:
    """Benchmark memory usage for different operations."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize memory benchmark.
        
        Args:
            device: Device to benchmark on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
    
    def benchmark_function(
        self,
        func: Callable,
        *args,
        runs: int = 5,
        warmup_runs: int = 2,
        label: str = "function",
        **kwargs
    ) -> Dict[str, Any]:
        """Benchmark memory usage of a function.
        
        Args:
            func: Function to benchmark
            *args: Function arguments
            runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            label: Label for the benchmark
            **kwargs: Function keyword arguments
            
        Returns:
            Benchmark results
        """
        # Warmup runs
        for _ in range(warmup_runs):
            force_garbage_collection()
            func(*args, **kwargs)
        
        # Actual benchmark runs
        run_results = []
        
        for run in range(runs):
            force_garbage_collection()
            
            with memory_profiler(self.device, f"{label}_run_{run}") as tracker:
                result = func(*args, **kwargs)
            
            run_data = {
                'run': run,
                'memory_tracking': tracker.result,
                'function_result': result
            }
            run_results.append(run_data)
        
        # Calculate statistics
        stats = self._calculate_benchmark_stats(run_results, label)
        
        benchmark_result = {
            'label': label,
            'runs': run_results,
            'statistics': stats,
            'device': str(self.device)
        }
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def _calculate_benchmark_stats(self, run_results: List[Dict], label: str) -> Dict[str, Any]:
        """Calculate statistics from benchmark runs."""
        if not run_results:
            return {}
        
        # Extract memory differences
        memory_diffs = []
        for run in run_results:
            diff = run['memory_tracking']['difference']
            memory_diffs.append(diff)
        
        # Calculate CPU memory stats
        cpu_stats = {}
        if memory_diffs and 'cpu_memory' in memory_diffs[0]:
            rss_diffs = [d['cpu_memory']['rss_mb_diff'] for d in memory_diffs]
            cpu_stats = {
                'avg_rss_mb_diff': sum(rss_diffs) / len(rss_diffs),
                'max_rss_mb_diff': max(rss_diffs),
                'min_rss_mb_diff': min(rss_diffs)
            }
        
        # Calculate GPU memory stats
        gpu_stats = {}
        if memory_diffs and 'gpu_memory' in memory_diffs[0]:
            gpu_data = memory_diffs[0]['gpu_memory']
            
            if isinstance(gpu_data, dict) and any('allocated_mb_diff' in v for v in gpu_data.values()):
                # Multiple GPUs
                for device in gpu_data:
                    allocated_diffs = [d['gpu_memory'][device]['allocated_mb_diff'] for d in memory_diffs if device in d['gpu_memory']]
                    if allocated_diffs:
                        gpu_stats[device] = {
                            'avg_allocated_mb_diff': sum(allocated_diffs) / len(allocated_diffs),
                            'max_allocated_mb_diff': max(allocated_diffs),
                            'min_allocated_mb_diff': min(allocated_diffs)
                        }
            else:
                # Single GPU
                allocated_diffs = [d['gpu_memory']['allocated_mb_diff'] for d in memory_diffs]
                gpu_stats = {
                    'avg_allocated_mb_diff': sum(allocated_diffs) / len(allocated_diffs),
                    'max_allocated_mb_diff': max(allocated_diffs),
                    'min_allocated_mb_diff': min(allocated_diffs)
                }
        
        return {
            'cpu_memory': cpu_stats,
            'gpu_memory': gpu_stats,
            'total_runs': len(run_results)
        }
    
    def compare_benchmarks(self, labels: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmarks.
        
        Args:
            labels: Labels of benchmarks to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'labels': labels,
            'comparisons': {}
        }
        
        # Find benchmarks with given labels
        benchmark_map = {}
        for result in self.results:
            if result['label'] in labels:
                benchmark_map[result['label']] = result
        
        # Compare memory usage
        for label in labels:
            if label in benchmark_map:
                stats = benchmark_map[label]['statistics']
                comparison['comparisons'][label] = stats
        
        return comparison
    
    def export_results(self, filepath: str):
        """Export benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        import json
        
        # Convert tensor data to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = self._make_serializable(result)
            serializable_results.append(serializable_result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results exported to {filepath}")
    
    def _make_serializable(self, obj):
        """Make object serializable by converting tensors to lists."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


def get_model_memory_footprint(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculate memory footprint of a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Memory footprint information
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / 1024 / 1024,
        'buffers_mb': buffer_size / 1024 / 1024,
        'total_mb': total_size / 1024 / 1024,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'buffer_count': sum(b.numel() for b in model.buffers())
    }
