"""Memory profiling utilities for chunked decomposition models."""

import torch
import psutil
import time
import gc
import threading
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from ..utils.memory_utils import MemoryTracker, force_garbage_collection

logger = logging.getLogger(__name__)


@dataclass
class MemoryProfile:
    """Container for memory profiling results."""
    peak_cpu_memory: float = 0.0  # MB
    peak_gpu_memory: float = 0.0  # MB
    avg_cpu_memory: float = 0.0   # MB
    avg_gpu_memory: float = 0.0   # MB
    memory_timeline: List[Dict[str, Any]] = field(default_factory=list)
    operation_memories: Dict[str, float] = field(default_factory=dict)
    total_duration: float = 0.0  # seconds
    compression_memory_saved: Optional[float] = None  # MB


class MemoryProfiler:
    """Advanced memory profiler for model operations."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        sampling_interval: float = 0.1,
        track_allocations: bool = True
    ):
        """Initialize memory profiler.
        
        Args:
            device: Device to profile
            sampling_interval: Memory sampling interval in seconds
            track_allocations: Whether to track individual allocations
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sampling_interval = sampling_interval
        self.track_allocations = track_allocations
        
        self.memory_tracker = MemoryTracker(device)
        self.is_profiling = False
        self.sampling_thread = None
        self.memory_timeline = []
        self.operation_markers = []
        self.start_time = None
        
    def start_profiling(self):
        """Start memory profiling."""
        if self.is_profiling:
            logger.warning("Profiling already started")
            return
        
        self.is_profiling = True
        self.start_time = time.time()
        self.memory_timeline.clear()
        self.operation_markers.clear()
        
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
        # Start sampling thread
        self.sampling_thread = threading.Thread(target=self._memory_sampling_loop)
        self.sampling_thread.daemon = True
        self.sampling_thread.start()
        
        # Enable CUDA memory tracking if available
        if torch.cuda.is_available() and self.track_allocations:
            torch.cuda.memory._record_memory_history(True)
        
        logger.info("Memory profiling started")
    
    def stop_profiling(self) -> MemoryProfile:
        """Stop memory profiling and return results.
        
        Returns:
            Memory profiling results
        """
        if not self.is_profiling:
            logger.warning("Profiling not started")
            return MemoryProfile()
        
        self.is_profiling = False
        
        # Stop memory tracking
        tracking_result = self.memory_tracker.stop_tracking()
        
        # Stop CUDA memory tracking
        if torch.cuda.is_available() and self.track_allocations:
            torch.cuda.memory._record_memory_history(False)
        
        # Wait for sampling thread to finish
        if self.sampling_thread:
            self.sampling_thread.join(timeout=1.0)
        
        # Calculate profile statistics
        profile = self._calculate_profile_stats()
        
        logger.info("Memory profiling stopped")
        return profile
    
    def _memory_sampling_loop(self):
        """Background loop for memory sampling."""
        while self.is_profiling:
            try:
                current_time = time.time() - self.start_time
                memory_info = self.memory_tracker.get_memory_info()
                memory_info['timestamp'] = current_time
                
                self.memory_timeline.append(memory_info)
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in memory sampling: {e}")
                break
    
    def mark_operation(self, operation_name: str):
        """Mark the start of an operation for memory tracking.
        
        Args:
            operation_name: Name of the operation
        """
        if not self.is_profiling:
            return
        
        current_time = time.time() - self.start_time
        memory_info = self.memory_tracker.get_memory_info()
        
        marker = {
            'operation': operation_name,
            'timestamp': current_time,
            'memory_info': memory_info
        }
        
        self.operation_markers.append(marker)
    
    def _calculate_profile_stats(self) -> MemoryProfile:
        """Calculate profiling statistics from collected data."""
        if not self.memory_timeline:
            return MemoryProfile()
        
        # Extract CPU and GPU memory values
        cpu_memories = []
        gpu_memories = []
        
        for sample in self.memory_timeline:
            if 'cpu_memory' in sample:
                cpu_memories.append(sample['cpu_memory']['rss_mb'])
            
            if 'gpu_memory' in sample:
                gpu_mem = sample['gpu_memory']
                if isinstance(gpu_mem, dict):
                    # Multiple GPUs - sum allocated memory
                    total_gpu = sum(mem['allocated_mb'] for mem in gpu_mem.values())
                    gpu_memories.append(total_gpu)
                else:
                    gpu_memories.append(gpu_mem['allocated_mb'])
        
        # Calculate statistics
        peak_cpu = max(cpu_memories) if cpu_memories else 0.0
        avg_cpu = np.mean(cpu_memories) if cpu_memories else 0.0
        peak_gpu = max(gpu_memories) if gpu_memories else 0.0
        avg_gpu = np.mean(gpu_memories) if gpu_memories else 0.0
        
        total_duration = self.memory_timeline[-1]['timestamp'] if self.memory_timeline else 0.0
        
        # Calculate operation-specific memory usage
        operation_memories = {}
        for i, marker in enumerate(self.operation_markers):
            op_name = marker['operation']
            op_memory = 0.0
            
            if 'cpu_memory' in marker['memory_info']:
                op_memory = marker['memory_info']['cpu_memory']['rss_mb']
            elif 'gpu_memory' in marker['memory_info']:
                gpu_mem = marker['memory_info']['gpu_memory']
                if isinstance(gpu_mem, dict):
                    op_memory = sum(mem['allocated_mb'] for mem in gpu_mem.values())
                else:
                    op_memory = gpu_mem['allocated_mb']
            
            operation_memories[op_name] = op_memory
        
        return MemoryProfile(
            peak_cpu_memory=peak_cpu,
            peak_gpu_memory=peak_gpu,
            avg_cpu_memory=avg_cpu,
            avg_gpu_memory=avg_gpu,
            memory_timeline=self.memory_timeline.copy(),
            operation_memories=operation_memories,
            total_duration=total_duration
        )
    
    def profile_model_forward(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        num_runs: int = 5
    ) -> Dict[str, MemoryProfile]:
        """Profile memory usage during model forward passes.
        
        Args:
            model: Model to profile
            inputs: Input tensor
            num_runs: Number of forward passes
            
        Returns:
            Dictionary of run_id -> memory profile
        """
        profiles = {}
        
        for run in range(num_runs):
            # Clean memory before each run
            force_garbage_collection()
            
            self.start_profiling()
            
            with torch.no_grad():
                self.mark_operation(f"forward_pass_{run}")
                outputs = model(inputs)
                self.mark_operation(f"forward_pass_{run}_end")
            
            profile = self.stop_profiling()
            profiles[f"run_{run}"] = profile
        
        return profiles
    
    def compare_memory_usage(
        self,
        models: Dict[str, torch.nn.Module],
        inputs: torch.Tensor,
        num_runs: int = 3
    ) -> Dict[str, MemoryProfile]:
        """Compare memory usage across multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            inputs: Input tensor
            num_runs: Number of runs per model
            
        Returns:
            Dictionary of model_name -> aggregated memory profile
        """
        model_profiles = {}
        
        for model_name, model in models.items():
            logger.info(f"Profiling memory for model: {model_name}")
            
            # Profile multiple runs
            run_profiles = self.profile_model_forward(model, inputs, num_runs)
            
            # Aggregate results
            aggregated = self._aggregate_profiles(list(run_profiles.values()))
            model_profiles[model_name] = aggregated
        
        return model_profiles
    
    def _aggregate_profiles(self, profiles: List[MemoryProfile]) -> MemoryProfile:
        """Aggregate multiple memory profiles.
        
        Args:
            profiles: List of memory profiles to aggregate
            
        Returns:
            Aggregated memory profile
        """
        if not profiles:
            return MemoryProfile()
        
        # Calculate averages
        peak_cpu = np.mean([p.peak_cpu_memory for p in profiles])
        peak_gpu = np.mean([p.peak_gpu_memory for p in profiles])
        avg_cpu = np.mean([p.avg_cpu_memory for p in profiles])
        avg_gpu = np.mean([p.avg_gpu_memory for p in profiles])
        total_duration = np.mean([p.total_duration for p in profiles])
        
        # Aggregate operation memories
        all_operations = set()
        for profile in profiles:
            all_operations.update(profile.operation_memories.keys())
        
        operation_memories = {}
        for op in all_operations:
            op_values = [p.operation_memories.get(op, 0.0) for p in profiles]
            operation_memories[op] = np.mean(op_values)
        
        return MemoryProfile(
            peak_cpu_memory=peak_cpu,
            peak_gpu_memory=peak_gpu,
            avg_cpu_memory=avg_cpu,
            avg_gpu_memory=avg_gpu,
            operation_memories=operation_memories,
            total_duration=total_duration
        )
    
    def plot_memory_timeline(
        self,
        profile: MemoryProfile,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot memory usage timeline.
        
        Args:
            profile: Memory profile to plot
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not profile.memory_timeline:
            logger.warning("No memory timeline data to plot")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extract timeline data
        timestamps = [sample['timestamp'] for sample in profile.memory_timeline]
        cpu_memories = []
        gpu_memories = []
        
        for sample in profile.memory_timeline:
            if 'cpu_memory' in sample:
                cpu_memories.append(sample['cpu_memory']['rss_mb'])
            else:
                cpu_memories.append(0)
            
            if 'gpu_memory' in sample:
                gpu_mem = sample['gpu_memory']
                if isinstance(gpu_mem, dict):
                    total_gpu = sum(mem['allocated_mb'] for mem in gpu_mem.values())
                    gpu_memories.append(total_gpu)
                else:
                    gpu_memories.append(gpu_mem['allocated_mb'])
            else:
                gpu_memories.append(0)
        
        # Plot CPU memory
        ax1.plot(timestamps, cpu_memories, 'b-', linewidth=2, label='CPU Memory')
        ax1.set_ylabel('CPU Memory (MB)')
        ax1.set_title('CPU Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot GPU memory
        if max(gpu_memories) > 0:
            ax2.plot(timestamps, gpu_memories, 'r-', linewidth=2, label='GPU Memory')
            ax2.set_ylabel('GPU Memory (MB)')
            ax2.set_title('GPU Memory Usage Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No GPU Memory Data', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('GPU Memory Usage Over Time')
        
        ax2.set_xlabel('Time (seconds)')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Memory timeline plot saved to {save_path}")
        
        return fig
    
    def export_profile(
        self,
        profile: MemoryProfile,
        filepath: str,
        format: str = 'json'
    ):
        """Export memory profile to file.
        
        Args:
            profile: Memory profile to export
            filepath: Output file path
            format: Export format ('json', 'csv')
        """
        from dataclasses import asdict
        import json
        import csv
        
        profile_dict = asdict(profile)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(profile_dict, f, indent=2, default=str)
        elif format == 'csv':
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write summary statistics
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Peak CPU Memory (MB)', profile.peak_cpu_memory])
                writer.writerow(['Peak GPU Memory (MB)', profile.peak_gpu_memory])
                writer.writerow(['Avg CPU Memory (MB)', profile.avg_cpu_memory])
                writer.writerow(['Avg GPU Memory (MB)', profile.avg_gpu_memory])
                writer.writerow(['Total Duration (s)', profile.total_duration])
                
                # Write operation memories
                writer.writerow([])
                writer.writerow(['Operation', 'Memory (MB)'])
                for op, memory in profile.operation_memories.items():
                    writer.writerow([op, memory])
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Memory profile exported to {filepath}")


@contextmanager
def memory_profiler(
    device: Optional[torch.device] = None,
    sampling_interval: float = 0.1
):
    """Context manager for memory profiling.
    
    Args:
        device: Device to profile
        sampling_interval: Memory sampling interval
        
    Usage:
        with memory_profiler() as profiler:
            # Your code here
            profiler.mark_operation("operation_name")
            output = model(input)
        
        profile = profiler.result
    """
    profiler = MemoryProfiler(device, sampling_interval)
    profiler.start_profiling()
    
    try:
        yield profiler
    finally:
        profiler.result = profiler.stop_profiling()


class CompressionMemoryAnalyzer:
    """Analyze memory savings from model compression."""
    
    def __init__(self):
        """Initialize compression memory analyzer."""
        self.profiler = MemoryProfiler()
    
    def analyze_compression_savings(
        self,
        original_model: torch.nn.Module,
        compressed_model: torch.nn.Module,
        inputs: torch.Tensor,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Analyze memory savings from compression.
        
        Args:
            original_model: Original uncompressed model
            compressed_model: Compressed model
            inputs: Input tensor for inference
            num_runs: Number of runs for averaging
            
        Returns:
            Compression analysis results
        """
        # Profile original model
        original_profiles = self.profiler.profile_model_forward(
            original_model, inputs, num_runs
        )
        original_agg = self.profiler._aggregate_profiles(
            list(original_profiles.values())
        )
        
        # Profile compressed model
        compressed_profiles = self.profiler.profile_model_forward(
            compressed_model, inputs, num_runs
        )
        compressed_agg = self.profiler._aggregate_profiles(
            list(compressed_profiles.values())
        )
        
        # Calculate savings
        cpu_memory_saved = original_agg.peak_cpu_memory - compressed_agg.peak_cpu_memory
        gpu_memory_saved = original_agg.peak_gpu_memory - compressed_agg.peak_gpu_memory
        
        cpu_savings_percent = (cpu_memory_saved / original_agg.peak_cpu_memory * 100) if original_agg.peak_cpu_memory > 0 else 0
        gpu_savings_percent = (gpu_memory_saved / original_agg.peak_gpu_memory * 100) if original_agg.peak_gpu_memory > 0 else 0
        
        return {
            'original_profile': original_agg,
            'compressed_profile': compressed_agg,
            'cpu_memory_saved_mb': cpu_memory_saved,
            'gpu_memory_saved_mb': gpu_memory_saved,
            'cpu_savings_percent': cpu_savings_percent,
            'gpu_savings_percent': gpu_savings_percent,
            'compression_effective': cpu_memory_saved > 0 or gpu_memory_saved > 0
        }
    
    def plot_compression_comparison(
        self,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot compression memory comparison.
        
        Args:
            analysis_results: Results from analyze_compression_savings
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        original = analysis_results['original_profile']
        compressed = analysis_results['compressed_profile']
        
        # CPU memory comparison
        cpu_data = [original.peak_cpu_memory, compressed.peak_cpu_memory]
        ax1.bar(['Original', 'Compressed'], cpu_data, color=['red', 'green'])
        ax1.set_ylabel('Peak CPU Memory (MB)')
        ax1.set_title('CPU Memory Usage Comparison')
        
        # Add savings annotation
        cpu_saved = analysis_results['cpu_memory_saved_mb']
        cpu_percent = analysis_results['cpu_savings_percent']
        ax1.text(0.5, max(cpu_data) * 0.8, f'Saved: {cpu_saved:.1f} MB\n({cpu_percent:.1f}%)',
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        # GPU memory comparison
        gpu_data = [original.peak_gpu_memory, compressed.peak_gpu_memory]
        ax2.bar(['Original', 'Compressed'], gpu_data, color=['red', 'green'])
        ax2.set_ylabel('Peak GPU Memory (MB)')
        ax2.set_title('GPU Memory Usage Comparison')
        
        # Add savings annotation
        gpu_saved = analysis_results['gpu_memory_saved_mb']
        gpu_percent = analysis_results['gpu_savings_percent']
        ax2.text(0.5, max(gpu_data) * 0.8, f'Saved: {gpu_saved:.1f} MB\n({gpu_percent:.1f}%)',
                ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue'))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Compression comparison plot saved to {save_path}")
        
        return fig


def profile_kv_cache_memory(
    model_with_cache,
    inputs: torch.Tensor,
    sequence_lengths: List[int]
) -> Dict[str, MemoryProfile]:
    """Profile memory usage of KV cache at different sequence lengths.
    
    Args:
        model_with_cache: Model with KV cache implementation
        inputs: Input tensor
        sequence_lengths: List of sequence lengths to test
        
    Returns:
        Dictionary of sequence_length -> memory profile
    """
    profiler = MemoryProfiler()
    profiles = {}
    
    for seq_len in sequence_lengths:
        # Prepare input of specific length
        if inputs.size(1) > seq_len:
            test_input = inputs[:, :seq_len]
        else:
            test_input = inputs
        
        logger.info(f"Profiling KV cache memory for sequence length: {seq_len}")
        
        # Clear cache and memory
        if hasattr(model_with_cache, 'clear_cache'):
            model_with_cache.clear_cache()
        force_garbage_collection()
        
        # Profile forward pass
        profiler.start_profiling()
        profiler.mark_operation(f"kv_cache_seq_len_{seq_len}")
        
        with torch.no_grad():
            outputs = model_with_cache(test_input)
        
        profile = profiler.stop_profiling()
        profiles[f"seq_len_{seq_len}"] = profile
    
    return profiles
