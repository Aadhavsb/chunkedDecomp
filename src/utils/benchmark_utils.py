"""Comprehensive benchmarking utilities for chunked decomposition models."""

import torch
import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from dataclasses import dataclass, field
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import csv

from .memory_utils import MemoryTracker, force_garbage_collection
from ..evaluation.performance_evaluator import PerformanceEvaluator, PerformanceMetrics
from ..evaluation.memory_profiler import MemoryProfiler, MemoryProfile

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    num_warmup_runs: int = 3
    num_benchmark_runs: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    compression_ratios: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    chunk_sizes: List[int] = field(default_factory=lambda: [64, 128, 256])
    compute_memory_profile: bool = True
    compute_performance_metrics: bool = True
    save_detailed_results: bool = True


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    config: BenchmarkConfig
    model_name: str
    device: str
    timestamp: str
    
    # Performance metrics
    inference_times: Dict[str, List[float]] = field(default_factory=dict)
    throughput: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, MemoryProfile] = field(default_factory=dict)
    performance_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    # Compression-specific metrics
    compression_ratios: Dict[str, float] = field(default_factory=dict)
    reconstruction_errors: Dict[str, float] = field(default_factory=dict)
    
    # Scaling analysis
    batch_size_scaling: Dict[int, Dict[str, float]] = field(default_factory=dict)
    sequence_length_scaling: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveBenchmark:
    """Comprehensive benchmarking suite for chunked decomposition models."""
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize comprehensive benchmark.
        
        Args:
            config: Benchmark configuration
            device: Device to run benchmarks on
        """
        self.config = config or BenchmarkConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.performance_evaluator = PerformanceEvaluator(self.device)
        self.memory_profiler = MemoryProfiler(self.device)
        self.memory_tracker = MemoryTracker(self.device)
        
        self.results = []
    
    def benchmark_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        dataloader: torch.utils.data.DataLoader,
        tokenizer: Any = None
    ) -> BenchmarkResult:
        """Run comprehensive benchmark on a model.
        
        Args:
            model: Model to benchmark
            model_name: Name identifier for the model
            dataloader: DataLoader with test data
            tokenizer: Optional tokenizer for generation metrics
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting comprehensive benchmark for model: {model_name}")
        
        # Initialize result container
        result = BenchmarkResult(
            config=self.config,
            model_name=model_name,
            device=str(self.device),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        model.eval()
        
        # 1. Basic performance evaluation
        if self.config.compute_performance_metrics:
            logger.info("Computing performance metrics...")
            perf_metrics = self.performance_evaluator.evaluate_model(
                model, dataloader, tokenizer,
                compute_generation_metrics=tokenizer is not None
            )
            result.performance_metrics['overall'] = perf_metrics
        
        # 2. Memory profiling
        if self.config.compute_memory_profile:
            logger.info("Profiling memory usage...")
            sample_batch = next(iter(dataloader))
            if isinstance(sample_batch, dict):
                sample_input = sample_batch['input_ids']
            else:
                sample_input = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
            
            memory_profiles = self.memory_profiler.profile_model_forward(
                model, sample_input.to(self.device), self.config.num_benchmark_runs
            )
            result.memory_usage = memory_profiles
        
        # 3. Batch size scaling analysis
        logger.info("Analyzing batch size scaling...")
        result.batch_size_scaling = self._benchmark_batch_scaling(model, dataloader)
        
        # 4. Sequence length scaling analysis
        logger.info("Analyzing sequence length scaling...")
        result.sequence_length_scaling = self._benchmark_sequence_scaling(model, dataloader)
        
        # 5. Inference time analysis
        logger.info("Benchmarking inference times...")
        result.inference_times = self._benchmark_inference_times(model, dataloader)
        
        # 6. Throughput analysis
        logger.info("Computing throughput metrics...")
        result.throughput = self._compute_throughput_metrics(model, dataloader)
        
        # 7. Generate summary statistics
        result.summary = self._generate_summary_stats(result)
        
        self.results.append(result)
        logger.info(f"Benchmark completed for model: {model_name}")
        
        return result
    
    def _benchmark_batch_scaling(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark model performance across different batch sizes.
        
        Args:
            model: Model to benchmark
            dataloader: DataLoader with test data
            
        Returns:
            Dictionary of batch_size -> metrics
        """
        scaling_results = {}
        
        # Get sample data
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, dict):
            sample_input = sample_batch['input_ids']
        else:
            sample_input = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        
        original_batch_size = sample_input.size(0)
        
        for batch_size in self.config.batch_sizes:
            if batch_size > original_batch_size:
                # Repeat samples to create larger batch
                repeat_factor = (batch_size + original_batch_size - 1) // original_batch_size
                test_input = sample_input.repeat(repeat_factor, 1)[:batch_size]
            else:
                test_input = sample_input[:batch_size]
            
            test_input = test_input.to(self.device)
            
            # Warmup
            for _ in range(self.config.num_warmup_runs):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Benchmark
            inference_times = []
            memory_usage = []
            
            for _ in range(self.config.num_benchmark_runs):
                force_garbage_collection()
                
                # Memory before
                mem_before = self.memory_tracker.get_memory_info()
                
                # Time inference
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(test_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Memory after
                mem_after = self.memory_tracker.get_memory_info()
                
                inference_times.append(end_time - start_time)
                
                # Calculate memory usage
                if 'gpu_memory' in mem_before and 'gpu_memory' in mem_after:
                    if isinstance(mem_after['gpu_memory'], dict):
                        mem_used = sum(mem_after['gpu_memory'][dev]['allocated_mb'] 
                                     for dev in mem_after['gpu_memory'])
                    else:
                        mem_used = mem_after['gpu_memory']['allocated_mb']
                    memory_usage.append(mem_used)
            
            # Calculate metrics
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            avg_memory = np.mean(memory_usage) if memory_usage else 0.0
            throughput = (batch_size * sample_input.size(1)) / avg_time if avg_time > 0 else 0
            
            scaling_results[batch_size] = {
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'avg_memory_usage': avg_memory,
                'throughput': throughput,
                'time_per_sample': avg_time / batch_size
            }
        
        return scaling_results
    
    def _benchmark_sequence_scaling(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark model performance across different sequence lengths.
        
        Args:
            model: Model to benchmark
            dataloader: DataLoader with test data
            
        Returns:
            Dictionary of sequence_length -> metrics
        """
        scaling_results = {}
        
        # Get sample data
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, dict):
            sample_input = sample_batch['input_ids']
        else:
            sample_input = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        
        batch_size = sample_input.size(0)
        original_seq_len = sample_input.size(1)
        
        for seq_len in self.config.sequence_lengths:
            if seq_len > original_seq_len:
                # Pad sequence to desired length
                pad_length = seq_len - original_seq_len
                test_input = torch.cat([
                    sample_input,
                    torch.zeros(batch_size, pad_length, dtype=sample_input.dtype)
                ], dim=1)
            else:
                test_input = sample_input[:, :seq_len]
            
            test_input = test_input.to(self.device)
            
            # Warmup
            for _ in range(self.config.num_warmup_runs):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Benchmark
            inference_times = []
            memory_usage = []
            
            for _ in range(self.config.num_benchmark_runs):
                force_garbage_collection()
                
                # Memory before
                mem_before = self.memory_tracker.get_memory_info()
                
                # Time inference
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(test_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # Memory after
                mem_after = self.memory_tracker.get_memory_info()
                
                inference_times.append(end_time - start_time)
                
                # Calculate memory usage
                if 'gpu_memory' in mem_before and 'gpu_memory' in mem_after:
                    if isinstance(mem_after['gpu_memory'], dict):
                        mem_used = sum(mem_after['gpu_memory'][dev]['allocated_mb'] 
                                     for dev in mem_after['gpu_memory'])
                    else:
                        mem_used = mem_after['gpu_memory']['allocated_mb']
                    memory_usage.append(mem_used)
            
            # Calculate metrics
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            avg_memory = np.mean(memory_usage) if memory_usage else 0.0
            throughput = (batch_size * seq_len) / avg_time if avg_time > 0 else 0
            
            scaling_results[seq_len] = {
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'avg_memory_usage': avg_memory,
                'throughput': throughput,
                'time_per_token': avg_time / (batch_size * seq_len)
            }
        
        return scaling_results
    
    def _benchmark_inference_times(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, List[float]]:
        """Benchmark detailed inference times.
        
        Args:
            model: Model to benchmark
            dataloader: DataLoader with test data
            
        Returns:
            Dictionary of timing measurements
        """
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, dict):
            sample_input = sample_batch['input_ids']
        else:
            sample_input = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        
        sample_input = sample_input.to(self.device)
        
        # Warmup
        for _ in range(self.config.num_warmup_runs):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Detailed timing
        forward_times = []
        first_token_times = []
        
        for _ in range(self.config.num_benchmark_runs):
            force_garbage_collection()
            
            # Full forward pass timing
            start_time = time.time()
            with torch.no_grad():
                outputs = model(sample_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            forward_times.append(end_time - start_time)
            
            # First token timing (if generation method exists)
            if hasattr(model, 'generate'):
                start_time = time.time()
                with torch.no_grad():
                    # Generate just one additional token
                    generated = model.generate(
                        sample_input,
                        max_new_tokens=1,
                        do_sample=False,
                        pad_token_id=0
                    )
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                first_token_times.append(end_time - start_time)
        
        return {
            'forward_pass': forward_times,
            'first_token_generation': first_token_times
        }
    
    def _compute_throughput_metrics(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Compute throughput metrics.
        
        Args:
            model: Model to benchmark
            dataloader: DataLoader with test data
            
        Returns:
            Dictionary of throughput metrics
        """
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, dict):
            sample_input = sample_batch['input_ids']
        else:
            sample_input = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        
        sample_input = sample_input.to(self.device)
        batch_size = sample_input.size(0)
        seq_length = sample_input.size(1)
        
        # Warmup
        for _ in range(self.config.num_warmup_runs):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Measure throughput
        total_time = 0.0
        total_tokens = 0
        
        for _ in range(self.config.num_benchmark_runs):
            start_time = time.time()
            with torch.no_grad():
                outputs = model(sample_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_tokens += (batch_size * seq_length)
        
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        batches_per_second = self.config.num_benchmark_runs / total_time if total_time > 0 else 0
        samples_per_second = (self.config.num_benchmark_runs * batch_size) / total_time if total_time > 0 else 0
        
        return {
            'tokens_per_second': tokens_per_second,
            'batches_per_second': batches_per_second,
            'samples_per_second': samples_per_second,
            'avg_time_per_batch': total_time / self.config.num_benchmark_runs
        }
    
    def _generate_summary_stats(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results.
        
        Args:
            result: Benchmark result to summarize
            
        Returns:
            Summary statistics
        """
        summary = {}
        
        # Performance metrics summary
        if result.performance_metrics:
            perf = result.performance_metrics['overall']
            summary['performance'] = {
                'accuracy': perf.accuracy,
                'perplexity': perf.perplexity,
                'inference_time': perf.inference_time,
                'throughput': perf.throughput
            }
        
        # Memory usage summary
        if result.memory_usage:
            memory_profiles = list(result.memory_usage.values())
            avg_peak_cpu = np.mean([p.peak_cpu_memory for p in memory_profiles])
            avg_peak_gpu = np.mean([p.peak_gpu_memory for p in memory_profiles])
            
            summary['memory'] = {
                'avg_peak_cpu_mb': avg_peak_cpu,
                'avg_peak_gpu_mb': avg_peak_gpu,
                'total_peak_memory_mb': avg_peak_cpu + avg_peak_gpu
            }
        
        # Scaling summary
        if result.batch_size_scaling:
            batch_throughputs = [metrics['throughput'] for metrics in result.batch_size_scaling.values()]
            summary['batch_scaling'] = {
                'min_throughput': min(batch_throughputs),
                'max_throughput': max(batch_throughputs),
                'throughput_range': max(batch_throughputs) - min(batch_throughputs)
            }
        
        if result.sequence_length_scaling:
            seq_throughputs = [metrics['throughput'] for metrics in result.sequence_length_scaling.values()]
            summary['sequence_scaling'] = {
                'min_throughput': min(seq_throughputs),
                'max_throughput': max(seq_throughputs),
                'throughput_range': max(seq_throughputs) - min(seq_throughputs)
            }
        
        # Inference time summary
        if result.inference_times:
            forward_times = result.inference_times.get('forward_pass', [])
            if forward_times:
                summary['inference_timing'] = {
                    'avg_forward_time': np.mean(forward_times),
                    'std_forward_time': np.std(forward_times),
                    'min_forward_time': np.min(forward_times),
                    'max_forward_time': np.max(forward_times)
                }
        
        # Throughput summary
        if result.throughput:
            summary['overall_throughput'] = result.throughput
        
        return summary
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        dataloader: torch.utils.data.DataLoader,
        tokenizer: Any = None
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple models using comprehensive benchmarks.
        
        Args:
            models: Dictionary of model_name -> model
            dataloader: DataLoader with test data
            tokenizer: Optional tokenizer for generation metrics
            
        Returns:
            Dictionary of model_name -> benchmark results
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Benchmarking model: {model_name}")
            result = self.benchmark_model(model, model_name, dataloader, tokenizer)
            comparison_results[model_name] = result
        
        return comparison_results
    
    def plot_comparison_results(
        self,
        comparison_results: Dict[str, BenchmarkResult],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison results across models.
        
        Args:
            comparison_results: Dictionary of model_name -> benchmark results
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Benchmark Comparison', fontsize=16)
        
        models = list(comparison_results.keys())
        
        # 1. Throughput comparison
        throughputs = []
        for model in models:
            result = comparison_results[model]
            if result.throughput:
                throughputs.append(result.throughput.get('tokens_per_second', 0))
            else:
                throughputs.append(0)
        
        axes[0, 0].bar(models, throughputs)
        axes[0, 0].set_title('Throughput (Tokens/Second)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Memory usage comparison
        memory_usage = []
        for model in models:
            result = comparison_results[model]
            if result.summary and 'memory' in result.summary:
                memory_usage.append(result.summary['memory']['total_peak_memory_mb'])
            else:
                memory_usage.append(0)
        
        axes[0, 1].bar(models, memory_usage)
        axes[0, 1].set_title('Peak Memory Usage (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Accuracy comparison
        accuracies = []
        for model in models:
            result = comparison_results[model]
            if result.performance_metrics and 'overall' in result.performance_metrics:
                accuracies.append(result.performance_metrics['overall'].accuracy)
            else:
                accuracies.append(0)
        
        axes[0, 2].bar(models, accuracies)
        axes[0, 2].set_title('Accuracy')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Batch size scaling
        for model in models:
            result = comparison_results[model]
            if result.batch_size_scaling:
                batch_sizes = list(result.batch_size_scaling.keys())
                throughputs = [result.batch_size_scaling[bs]['throughput'] for bs in batch_sizes]
                axes[1, 0].plot(batch_sizes, throughputs, marker='o', label=model)
        
        axes[1, 0].set_title('Batch Size Scaling')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Throughput')
        axes[1, 0].legend()
        
        # 5. Sequence length scaling
        for model in models:
            result = comparison_results[model]
            if result.sequence_length_scaling:
                seq_lengths = list(result.sequence_length_scaling.keys())
                throughputs = [result.sequence_length_scaling[sl]['throughput'] for sl in seq_lengths]
                axes[1, 1].plot(seq_lengths, throughputs, marker='o', label=model)
        
        axes[1, 1].set_title('Sequence Length Scaling')
        axes[1, 1].set_xlabel('Sequence Length')
        axes[1, 1].set_ylabel('Throughput')
        axes[1, 1].legend()
        
        # 6. Inference time distribution
        for model in models:
            result = comparison_results[model]
            if result.inference_times and 'forward_pass' in result.inference_times:
                times = result.inference_times['forward_pass']
                axes[1, 2].hist(times, alpha=0.6, label=model, bins=20)
        
        axes[1, 2].set_title('Inference Time Distribution')
        axes[1, 2].set_xlabel('Time (seconds)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        return fig
    
    def export_results(
        self,
        results: Dict[str, BenchmarkResult],
        output_dir: str,
        format: str = 'json'
    ):
        """Export benchmark results to files.
        
        Args:
            results: Dictionary of model_name -> benchmark results
            output_dir: Output directory path
            format: Export format ('json', 'csv', 'both')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, result in results.items():
            if format in ['json', 'both']:
                json_path = output_path / f"{model_name}_benchmark.json"
                self._export_json(result, json_path)
            
            if format in ['csv', 'both']:
                csv_path = output_path / f"{model_name}_benchmark.csv"
                self._export_csv(result, csv_path)
        
        logger.info(f"Benchmark results exported to {output_dir}")
    
    def _export_json(self, result: BenchmarkResult, filepath: Path):
        """Export result to JSON format."""
        from dataclasses import asdict
        
        # Convert result to dictionary, handling non-serializable objects
        result_dict = asdict(result)
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
    
    def _export_csv(self, result: BenchmarkResult, filepath: Path):
        """Export result to CSV format."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write summary statistics
            writer.writerow(['Benchmark Summary', result.model_name])
            writer.writerow(['Timestamp', result.timestamp])
            writer.writerow(['Device', result.device])
            writer.writerow([])
            
            # Write summary metrics
            if result.summary:
                writer.writerow(['Summary Metrics'])
                for category, metrics in result.summary.items():
                    writer.writerow([f"{category.title()} Metrics"])
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            writer.writerow([metric, value])
                    writer.writerow([])
            
            # Write scaling results
            if result.batch_size_scaling:
                writer.writerow(['Batch Size Scaling'])
                writer.writerow(['Batch Size', 'Avg Time', 'Throughput', 'Memory Usage'])
                for batch_size, metrics in result.batch_size_scaling.items():
                    writer.writerow([
                        batch_size,
                        metrics['avg_inference_time'],
                        metrics['throughput'],
                        metrics['avg_memory_usage']
                    ])
                writer.writerow([])
            
            if result.sequence_length_scaling:
                writer.writerow(['Sequence Length Scaling'])
                writer.writerow(['Sequence Length', 'Avg Time', 'Throughput', 'Memory Usage'])
                for seq_len, metrics in result.sequence_length_scaling.items():
                    writer.writerow([
                        seq_len,
                        metrics['avg_inference_time'],
                        metrics['throughput'],
                        metrics['avg_memory_usage']
                    ])


def quick_benchmark(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """Quick benchmark for rapid testing.
    
    Args:
        model: Model to benchmark
        input_tensor: Input tensor for inference
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Quick benchmark metrics
    """
    model.eval()
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Benchmark
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    batch_size = input_tensor.size(0)
    seq_length = input_tensor.size(1)
    total_tokens = batch_size * seq_length
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = total_tokens / avg_time if avg_time > 0 else 0
    
    return {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'min_inference_time': np.min(times),
        'max_inference_time': np.max(times),
        'throughput_tokens_per_sec': throughput,
        'time_per_token': avg_time / total_tokens if total_tokens > 0 else 0
    }
