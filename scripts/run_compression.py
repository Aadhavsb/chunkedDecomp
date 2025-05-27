#!/usr/bin/env python3
"""Main script for running model compression with chunked decomposition."""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
import torch
import yaml
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.compressed_model import create_compressed_model, CompressedModelWrapper
from utils.data_utils import DatasetManager, create_dataloader_from_config
from evaluation.performance_evaluator import PerformanceEvaluator
from evaluation.memory_profiler import MemoryProfiler
from utils.memory_utils import force_garbage_collection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('compression.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def setup_device(device_name: str = 'auto') -> torch.device:
    """Setup computation device.
    
    Args:
        device_name: Device name ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        PyTorch device
    """
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device


def run_compression(
    model_config: Dict[str, Any],
    compression_config: Dict[str, Any],
    data_config: Dict[str, Any],
    output_dir: str,
    device: torch.device,
    evaluate_performance: bool = True,
    profile_memory: bool = True
) -> Dict[str, Any]:
    """Run model compression with evaluation.
    
    Args:
        model_config: Model configuration
        compression_config: Compression configuration
        data_config: Data configuration
        output_dir: Output directory for results
        device: Computation device
        evaluate_performance: Whether to evaluate performance
        profile_memory: Whether to profile memory usage
        
    Returns:
        Compression results
    """
    logger.info("Starting model compression...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset for evaluation
    dataset_manager = DatasetManager()
    dataloader = create_dataloader_from_config(data_config)
    
    logger.info(f"Dataset loaded: {len(dataloader)} batches")
    
    # Create compressed model
    start_time = time.time()
    
    compressed_model = create_compressed_model(
        model_name_or_path=model_config['name_or_path'],
        compression_config=compression_config,
        device=device,
        model_kwargs=model_config.get('model_kwargs', {})
    )
    
    compression_time = time.time() - start_time
    
    logger.info(f"Model compression completed in {compression_time:.2f} seconds")
    logger.info(f"Compression ratio: {compressed_model.compression_stats.compression_ratio:.2f}x")
    logger.info(f"Memory saved: {compressed_model.compression_stats.memory_saved_mb:.1f} MB")
    
    # Collect results
    results = {
        'model_config': model_config,
        'compression_config': compression_config,
        'compression_info': compressed_model.get_compression_info(),
        'compression_time': compression_time,
        'device': str(device),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Evaluate performance
    if evaluate_performance:
        logger.info("Evaluating model performance...")
        
        evaluator = PerformanceEvaluator(device)
        
        # Evaluate compressed model
        performance_metrics = evaluator.evaluate_model(
            model=compressed_model.model,
            dataloader=dataloader,
            max_batches=data_config.get('max_eval_batches', 50)
        )
        
        results['performance_metrics'] = {
            'accuracy': performance_metrics.accuracy,
            'perplexity': performance_metrics.perplexity,
            'inference_time': performance_metrics.inference_time,
            'throughput': performance_metrics.throughput
        }
        
        logger.info(f"Performance - Accuracy: {performance_metrics.accuracy:.4f}, "
                   f"Perplexity: {performance_metrics.perplexity:.2f}")
        
        # Benchmark inference speed
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, dict):
            sample_input = sample_batch['input_ids']
        else:
            sample_input = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        
        benchmark_results = compressed_model.benchmark_performance(
            sample_input.to(device),
            num_runs=compression_config.get('benchmark_runs', 10)
        )
        
        results['benchmark_results'] = benchmark_results
    
    # Profile memory usage
    if profile_memory:
        logger.info("Profiling memory usage...")
        
        profiler = MemoryProfiler(device)
        
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, dict):
            sample_input = sample_batch['input_ids']
        else:
            sample_input = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        
        memory_profiles = profiler.profile_model_forward(
            model=compressed_model.model,
            inputs=sample_input.to(device),
            num_runs=5
        )
        
        # Aggregate memory profiles
        avg_profile = profiler._aggregate_profiles(list(memory_profiles.values()))
        
        results['memory_profile'] = {
            'peak_cpu_memory': avg_profile.peak_cpu_memory,
            'peak_gpu_memory': avg_profile.peak_gpu_memory,
            'avg_cpu_memory': avg_profile.avg_cpu_memory,
            'avg_gpu_memory': avg_profile.avg_gpu_memory
        }
        
        # Get current memory usage
        memory_usage = compressed_model.get_memory_usage()
        results['current_memory_usage'] = memory_usage
        
        logger.info(f"Memory usage - Peak GPU: {avg_profile.peak_gpu_memory:.1f} MB, "
                   f"Model: {memory_usage['model_memory_mb']:.1f} MB")
    
    # Save compressed model
    model_save_path = output_path / 'compressed_model'
    compressed_model.save_compressed_model(str(model_save_path))
    logger.info(f"Compressed model saved to: {model_save_path}")
    
    # Save results
    results_path = output_path / 'compression_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Save layer-wise compression details
    layer_details = compressed_model.get_layer_compression_details()
    layer_details_path = output_path / 'layer_compression_details.json'
    with open(layer_details_path, 'w') as f:
        json.dump(layer_details, f, indent=2, default=str)
    
    return results


def run_comparison_study(
    model_config: Dict[str, Any],
    compression_configs: Dict[str, Dict[str, Any]],
    data_config: Dict[str, Any],
    output_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    """Run compression comparison study with multiple configurations.
    
    Args:
        model_config: Model configuration
        compression_configs: Dictionary of compression configurations
        data_config: Data configuration
        output_dir: Output directory for results
        device: Computation device
        
    Returns:
        Comparison results
    """
    logger.info(f"Starting comparison study with {len(compression_configs)} configurations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {}
    
    for config_name, compression_config in compression_configs.items():
        logger.info(f"Running compression with config: {config_name}")
        
        try:
            # Run compression for this config
            config_output_dir = output_path / config_name
            
            results = run_compression(
                model_config=model_config,
                compression_config=compression_config,
                data_config=data_config,
                output_dir=str(config_output_dir),
                device=device,
                evaluate_performance=True,
                profile_memory=True
            )
            
            comparison_results[config_name] = results
            
            # Clean up GPU memory between runs
            force_garbage_collection()
            
        except Exception as e:
            logger.error(f"Failed to run compression with config {config_name}: {e}")
            comparison_results[config_name] = {'error': str(e)}
    
    # Save comparison results
    comparison_path = output_path / 'comparison_results.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    logger.info(f"Comparison study completed. Results saved to: {comparison_path}")
    
    # Generate comparison summary
    summary = generate_comparison_summary(comparison_results)
    summary_path = output_path / 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return comparison_results


def generate_comparison_summary(comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of comparison results.
    
    Args:
        comparison_results: Results from comparison study
        
    Returns:
        Summary statistics
    """
    summary = {
        'num_configs': len(comparison_results),
        'successful_configs': 0,
        'failed_configs': 0,
        'best_compression_ratio': {'config': None, 'ratio': 0},
        'best_performance': {'config': None, 'accuracy': 0},
        'lowest_memory': {'config': None, 'memory_mb': float('inf')},
        'config_rankings': []
    }
    
    valid_results = []
    
    for config_name, results in comparison_results.items():
        if 'error' in results:
            summary['failed_configs'] += 1
        else:
            summary['successful_configs'] += 1
            valid_results.append((config_name, results))
            
            # Track best metrics
            compression_ratio = results['compression_info']['compression_stats']['compression_ratio']
            if compression_ratio > summary['best_compression_ratio']['ratio']:
                summary['best_compression_ratio'] = {
                    'config': config_name,
                    'ratio': compression_ratio
                }
            
            if 'performance_metrics' in results:
                accuracy = results['performance_metrics']['accuracy']
                if accuracy > summary['best_performance']['accuracy']:
                    summary['best_performance'] = {
                        'config': config_name,
                        'accuracy': accuracy
                    }
            
            if 'memory_profile' in results:
                memory = results['memory_profile']['peak_gpu_memory']
                if memory < summary['lowest_memory']['memory_mb']:
                    summary['lowest_memory'] = {
                        'config': config_name,
                        'memory_mb': memory
                    }
    
    # Rank configurations by overall score (weighted combination of metrics)
    for config_name, results in valid_results:
        score = 0
        
        # Compression ratio (30% weight)
        compression_ratio = results['compression_info']['compression_stats']['compression_ratio']
        score += compression_ratio * 0.3
        
        # Performance (40% weight)
        if 'performance_metrics' in results:
            accuracy = results['performance_metrics']['accuracy']
            score += accuracy * 0.4
        
        # Memory efficiency (30% weight)
        if 'memory_profile' in results:
            memory_mb = results['memory_profile']['peak_gpu_memory']
            # Normalize memory score (lower is better)
            max_memory = max(r['memory_profile']['peak_gpu_memory'] 
                           for _, r in valid_results if 'memory_profile' in r)
            memory_score = (max_memory - memory_mb) / max_memory if max_memory > 0 else 0
            score += memory_score * 0.3
        
        summary['config_rankings'].append({
            'config': config_name,
            'overall_score': score,
            'compression_ratio': compression_ratio,
            'accuracy': results.get('performance_metrics', {}).get('accuracy', 0),
            'memory_mb': results.get('memory_profile', {}).get('peak_gpu_memory', 0)
        })
    
    # Sort by overall score
    summary['config_rankings'].sort(key=lambda x: x['overall_score'], reverse=True)
    
    return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run model compression with chunked decomposition')
    
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--compression_config', type=str, required=True,
                       help='Path to compression configuration file')
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data configuration file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, etc.)')
    parser.add_argument('--comparison_study', action='store_true',
                       help='Run comparison study with multiple compression configs')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip performance evaluation')
    parser.add_argument('--no_profile', action='store_true',
                       help='Skip memory profiling')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    
    if args.comparison_study:
        # Load multiple compression configs for comparison
        compression_configs = load_config(args.compression_config)
        
        if not isinstance(compression_configs, dict):
            raise ValueError("Compression config must be a dictionary for comparison study")
        
        results = run_comparison_study(
            model_config=model_config,
            compression_configs=compression_configs,
            data_config=data_config,
            output_dir=args.output_dir,
            device=device
        )
    else:
        # Single compression run
        compression_config = load_config(args.compression_config)
        
        results = run_compression(
            model_config=model_config,
            compression_config=compression_config,
            data_config=data_config,
            output_dir=args.output_dir,
            device=device,
            evaluate_performance=not args.no_eval,
            profile_memory=not args.no_profile
        )
    
    logger.info("Compression completed successfully!")
    
    # Print summary
    if args.comparison_study and 'comparison_summary' in results:
        summary = results['comparison_summary']
        logger.info(f"Best compression ratio: {summary['best_compression_ratio']['config']} "
                   f"({summary['best_compression_ratio']['ratio']:.2f}x)")
        logger.info(f"Best performance: {summary['best_performance']['config']} "
                   f"({summary['best_performance']['accuracy']:.4f})")
        logger.info(f"Lowest memory: {summary['lowest_memory']['config']} "
                   f"({summary['lowest_memory']['memory_mb']:.1f} MB)")
    else:
        if 'compression_info' in results:
            stats = results['compression_info']['compression_stats']
            logger.info(f"Final compression ratio: {stats['compression_ratio']:.2f}x")
            logger.info(f"Memory saved: {stats['memory_saved_mb']:.1f} MB")
            logger.info(f"Reconstruction error: {stats['reconstruction_error']:.6f}")


if __name__ == '__main__':
    main()
