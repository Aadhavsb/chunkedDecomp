#!/usr/bin/env python3
"""
Memory Benchmark Script for ChunkedDecomp

This script provides dedicated memory benchmarking capabilities including:
- Memory usage profiling across different model configurations
- Memory scaling analysis with batch size and sequence length
- Compression ratio vs memory trade-off analysis
- Memory efficiency comparison between models
- Peak memory detection and optimization

Usage:
    python benchmark_memory.py --model_name gpt2 --output_dir memory_results/
    python benchmark_memory.py --config configs/compression_configs.yaml --compare_compression
    python benchmark_memory.py --scaling_analysis --max_batch_size 32 --max_seq_length 1024
"""

import argparse
import os
import sys
import time
import yaml
import torch
import logging
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.chunked_decomp import ChunkedDecomp
from models.compressed_model import CompressedModelWrapper
from evaluation.memory_profiler import MemoryProfiler
from utils.memory_utils import MemoryTracker
from utils.data_utils import DatasetManager
from utils.benchmark_utils import ComprehensiveBenchmark

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('memory_benchmark.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def create_test_model(model_name: str, device: str = 'auto') -> torch.nn.Module:
    """Create a test model for benchmarking."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Creating test model: {model_name}")
    
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        return model
    except Exception as e:
        logging.error(f"Failed to create model {model_name}: {e}")
        raise

def memory_scaling_analysis(
    model_name: str,
    batch_sizes: List[int],
    sequence_lengths: List[int],
    device: str = 'auto',
    output_dir: str = 'memory_results'
) -> Dict[str, Any]:
    """Analyze memory scaling with batch size and sequence length."""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Running memory scaling analysis for {model_name}")
    
    results = {
        'model_name': model_name,
        'device': device,
        'scaling_data': [],
        'peak_memory': {},
        'oom_threshold': {}
    }
    
    # Create base model
    model = create_test_model(model_name, device)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            logging.info(f"Testing batch_size={batch_size}, seq_length={seq_length}")
            
            try:
                # Clear cache before each test
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Create test input
                dummy_text = "This is a test sentence. " * (seq_length // 6)
                inputs = tokenizer(
                    [dummy_text] * batch_size,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=seq_length
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Start memory tracking
                with MemoryTracker(device=device) as tracker:
                    model.eval()
                    with torch.no_grad():
                        # Forward pass
                        outputs = model(**inputs)
                        
                        # Optional generation test (for smaller configs)
                        if batch_size <= 4 and seq_length <= 256:
                            try:
                                generated = model.generate(
                                    inputs['input_ids'][:1],  # Single sample for generation
                                    max_length=min(seq_length + 50, 512),
                                    do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id
                                )
                            except Exception as gen_e:
                                logging.warning(f"Generation failed: {gen_e}")
                
                memory_stats = tracker.get_stats()
                
                scaling_point = {
                    'batch_size': batch_size,
                    'sequence_length': seq_length,
                    'peak_memory_mb': memory_stats.get('peak_memory_mb', 0),
                    'memory_allocated_mb': memory_stats.get('memory_allocated_mb', 0),
                    'memory_reserved_mb': memory_stats.get('memory_reserved_mb', 0),
                    'success': True,
                    'error': None
                }
                
                results['scaling_data'].append(scaling_point)
                
                logging.info(f"Success - Peak memory: {memory_stats.get('peak_memory_mb', 0):.2f} MB")
                
            except torch.cuda.OutOfMemoryError as e:
                logging.warning(f"OOM at batch_size={batch_size}, seq_length={seq_length}")
                
                # Record OOM point
                scaling_point = {
                    'batch_size': batch_size,
                    'sequence_length': seq_length,
                    'peak_memory_mb': -1,
                    'memory_allocated_mb': -1,
                    'memory_reserved_mb': -1,
                    'success': False,
                    'error': 'OutOfMemoryError'
                }
                
                results['scaling_data'].append(scaling_point)
                
                # Clear memory and continue
                if device == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error at batch_size={batch_size}, seq_length={seq_length}: {e}")
                
                scaling_point = {
                    'batch_size': batch_size,
                    'sequence_length': seq_length,
                    'peak_memory_mb': -1,
                    'memory_allocated_mb': -1,
                    'memory_reserved_mb': -1,
                    'success': False,
                    'error': str(e)
                }
                
                results['scaling_data'].append(scaling_point)
    
    # Analysis of results
    successful_runs = [p for p in results['scaling_data'] if p['success']]
    
    if successful_runs:
        # Find peak memory for each batch size
        for bs in batch_sizes:
            bs_runs = [p for p in successful_runs if p['batch_size'] == bs]
            if bs_runs:
                peak = max(bs_runs, key=lambda x: x['peak_memory_mb'])
                results['peak_memory'][bs] = peak
        
        # Find OOM thresholds
        for bs in batch_sizes:
            bs_runs = [p for p in results['scaling_data'] if p['batch_size'] == bs]
            oom_runs = [p for p in bs_runs if not p['success'] and p['error'] == 'OutOfMemoryError']
            if oom_runs:
                min_oom_seq = min(oom_runs, key=lambda x: x['sequence_length'])
                results['oom_threshold'][bs] = min_oom_seq['sequence_length']
    
    return results

def compression_memory_analysis(
    model_name: str,
    compression_configs: List[Dict[str, Any]],
    device: str = 'auto',
    output_dir: str = 'memory_results'
) -> Dict[str, Any]:
    """Analyze memory usage with different compression configurations."""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Running compression memory analysis for {model_name}")
    
    results = {
        'model_name': model_name,
        'device': device,
        'compression_results': [],
        'baseline_memory': None
    }
    
    # Test baseline (uncompressed) model
    logging.info("Testing baseline uncompressed model")
    baseline_model = create_test_model(model_name, device)
    
    test_batch_size = 4
    test_seq_length = 256
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create test input
    dummy_text = "This is a test sentence for memory benchmarking. " * 10
    test_inputs = tokenizer(
        [dummy_text] * test_batch_size,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=test_seq_length
    )
    test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
    
    # Baseline memory measurement
    with MemoryTracker(device=device) as tracker:
        baseline_model.eval()
        with torch.no_grad():
            outputs = baseline_model(**test_inputs)
    
    baseline_stats = tracker.get_stats()
    results['baseline_memory'] = baseline_stats
    
    logging.info(f"Baseline memory: {baseline_stats.get('peak_memory_mb', 0):.2f} MB")
    
    # Test compressed models
    for i, config in enumerate(compression_configs):
        logging.info(f"Testing compression config {i+1}/{len(compression_configs)}")
        
        try:
            # Clear memory
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Create compressed model
            chunked_decomp = ChunkedDecomp(
                model=create_test_model(model_name, device),
                config=config
            )
            
            # Apply compression
            chunked_decomp.apply_compression()
            compressed_model = chunked_decomp.model
            
            # Measure compressed model memory
            with MemoryTracker(device=device) as tracker:
                compressed_model.eval()
                with torch.no_grad():
                    outputs = compressed_model(**test_inputs)
            
            compressed_stats = tracker.get_stats()
            
            # Calculate compression metrics
            baseline_memory = baseline_stats.get('peak_memory_mb', 0)
            compressed_memory = compressed_stats.get('peak_memory_mb', 0)
            memory_reduction = (baseline_memory - compressed_memory) / baseline_memory * 100
            
            compression_result = {
                'config_index': i,
                'config': config,
                'memory_stats': compressed_stats,
                'memory_reduction_percent': memory_reduction,
                'compression_ratio': config.get('compression_ratio', 0.5),
                'chunk_size': config.get('chunk_size', 64),
                'success': True,
                'error': None
            }
            
            results['compression_results'].append(compression_result)
            
            logging.info(f"Compressed memory: {compressed_memory:.2f} MB "
                        f"(reduction: {memory_reduction:.1f}%)")
            
        except Exception as e:
            logging.error(f"Error with compression config {i}: {e}")
            
            compression_result = {
                'config_index': i,
                'config': config,
                'memory_stats': {},
                'memory_reduction_percent': 0,
                'compression_ratio': config.get('compression_ratio', 0.5),
                'chunk_size': config.get('chunk_size', 64),
                'success': False,
                'error': str(e)
            }
            
            results['compression_results'].append(compression_result)
    
    return results

def detailed_memory_profiling(
    model_name: str,
    operations: List[str],
    device: str = 'auto',
    output_dir: str = 'memory_results'
) -> Dict[str, Any]:
    """Detailed memory profiling of different operations."""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info(f"Running detailed memory profiling for {model_name}")
    
    model = create_test_model(model_name, device)
    profiler = MemoryProfiler(device=device, sampling_interval=0.1)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare test data
    test_text = "This is a comprehensive test for memory profiling of transformer models."
    test_inputs = tokenizer(
        test_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
    
    results = {
        'model_name': model_name,
        'device': device,
        'operation_profiles': {}
    }
    
    # Start profiling
    profiler.start_profiling()
    
    model.eval()
    
    for operation in operations:
        logging.info(f"Profiling operation: {operation}")
        
        try:
            profiler.mark_operation(f"{operation}_start")
            
            if operation == 'forward_pass':
                with torch.no_grad():
                    outputs = model(**test_inputs)
                    
            elif operation == 'generation':
                with torch.no_grad():
                    generated = model.generate(
                        test_inputs['input_ids'],
                        max_length=200,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
            elif operation == 'gradient_computation':
                model.train()
                outputs = model(**test_inputs, labels=test_inputs['input_ids'])
                loss = outputs.loss
                loss.backward()
                model.zero_grad()
                model.eval()
                
            elif operation == 'large_batch':
                large_inputs = tokenizer(
                    [test_text] * 16,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                large_inputs = {k: v.to(device) for k, v in large_inputs.items()}
                with torch.no_grad():
                    outputs = model(**large_inputs)
                    
            elif operation == 'long_sequence':
                long_text = test_text * 20
                long_inputs = tokenizer(
                    long_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                long_inputs = {k: v.to(device) for k, v in long_inputs.items()}
                with torch.no_grad():
                    outputs = model(**long_inputs)
            
            profiler.mark_operation(f"{operation}_end")
            
            # Clear memory between operations
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error in operation {operation}: {e}")
            profiler.mark_operation(f"{operation}_error")
    
    # Stop profiling and get results
    profile_data = profiler.stop_profiling()
    results['profile_data'] = profile_data
    
    # Analyze operation-specific memory usage
    timeline = profile_data.get('timeline', [])
    operations_analysis = {}
    
    for operation in operations:
        start_marker = f"{operation}_start"
        end_marker = f"{operation}_end"
        
        start_time = None
        end_time = None
        
        for entry in timeline:
            if entry.get('operation') == start_marker:
                start_time = entry['timestamp']
            elif entry.get('operation') == end_marker:
                end_time = entry['timestamp']
                break
        
        if start_time is not None and end_time is not None:
            # Find memory usage during operation
            operation_timeline = [
                entry for entry in timeline
                if start_time <= entry['timestamp'] <= end_time
            ]
            
            if operation_timeline:
                peak_memory = max(entry['memory_used'] for entry in operation_timeline)
                avg_memory = np.mean([entry['memory_used'] for entry in operation_timeline])
                
                operations_analysis[operation] = {
                    'peak_memory_mb': peak_memory,
                    'average_memory_mb': avg_memory,
                    'duration_seconds': end_time - start_time,
                    'memory_growth': peak_memory - operation_timeline[0]['memory_used']
                }
    
    results['operations_analysis'] = operations_analysis
    
    return results

def create_memory_plots(results: Dict[str, Any], output_dir: str):
    """Create memory benchmark visualization plots."""
    
    plt.style.use('seaborn-v0_8')
    os.makedirs(output_dir, exist_ok=True)
    
    # Scaling analysis plots
    if 'scaling_data' in results:
        scaling_data = results['scaling_data']
        successful_data = [p for p in scaling_data if p['success']]
        
        if successful_data:
            df = pd.DataFrame(successful_data)
            
            # Memory vs batch size
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Memory Scaling Analysis - {results.get("model_name", "Model")}', fontsize=16)
            
            # Pivot for heatmap
            pivot_data = df.pivot(index='sequence_length', columns='batch_size', values='peak_memory_mb')
            
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0])
            axes[0, 0].set_title('Peak Memory (MB) Heatmap')
            
            # Memory vs batch size for different sequence lengths
            for seq_len in df['sequence_length'].unique():
                seq_data = df[df['sequence_length'] == seq_len]
                axes[0, 1].plot(seq_data['batch_size'], seq_data['peak_memory_mb'], 
                               marker='o', label=f'seq_len={seq_len}')
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Peak Memory (MB)')
            axes[0, 1].set_title('Memory vs Batch Size')
            axes[0, 1].legend()
            
            # Memory vs sequence length
            for bs in df['batch_size'].unique():
                bs_data = df[df['batch_size'] == bs]
                axes[1, 0].plot(bs_data['sequence_length'], bs_data['peak_memory_mb'], 
                               marker='o', label=f'batch_size={bs}')
            axes[1, 0].set_xlabel('Sequence Length')
            axes[1, 0].set_ylabel('Peak Memory (MB)')
            axes[1, 0].set_title('Memory vs Sequence Length')
            axes[1, 0].legend()
            
            # Memory efficiency (memory per token)
            df['tokens_per_sample'] = df['sequence_length']
            df['total_tokens'] = df['batch_size'] * df['tokens_per_sample']
            df['memory_per_token'] = df['peak_memory_mb'] / df['total_tokens']
            
            scatter = axes[1, 1].scatter(df['total_tokens'], df['memory_per_token'], 
                                       c=df['batch_size'], cmap='viridis', alpha=0.7)
            axes[1, 1].set_xlabel('Total Tokens')
            axes[1, 1].set_ylabel('Memory per Token (MB)')
            axes[1, 1].set_title('Memory Efficiency')
            plt.colorbar(scatter, ax=axes[1, 1], label='Batch Size')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'memory_scaling_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # Compression analysis plots
    if 'compression_results' in results:
        compression_data = results['compression_results']
        successful_compression = [r for r in compression_data if r['success']]
        
        if successful_compression and results.get('baseline_memory'):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Compression Memory Analysis', fontsize=16)
            
            # Extract data
            compression_ratios = [r['compression_ratio'] for r in successful_compression]
            memory_reductions = [r['memory_reduction_percent'] for r in successful_compression]
            chunk_sizes = [r['chunk_size'] for r in successful_compression]
            
            baseline_memory = results['baseline_memory'].get('peak_memory_mb', 0)
            compressed_memories = [
                r['memory_stats'].get('peak_memory_mb', 0) 
                for r in successful_compression
            ]
            
            # Compression ratio vs memory reduction
            axes[0].scatter(compression_ratios, memory_reductions, c=chunk_sizes, 
                           cmap='plasma', alpha=0.7, s=100)
            axes[0].set_xlabel('Compression Ratio')
            axes[0].set_ylabel('Memory Reduction (%)')
            axes[0].set_title('Compression Ratio vs Memory Reduction')
            
            # Memory usage comparison
            x_pos = range(len(successful_compression) + 1)
            memory_values = [baseline_memory] + compressed_memories
            labels = ['Baseline'] + [f'Config {i}' for i in range(len(successful_compression))]
            
            bars = axes[1].bar(x_pos, memory_values, alpha=0.7)
            bars[0].set_color('red')  # Baseline in red
            axes[1].set_xlabel('Configuration')
            axes[1].set_ylabel('Peak Memory (MB)')
            axes[1].set_title('Memory Usage Comparison')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(labels, rotation=45)
            
            # Efficiency plot (compression ratio vs memory/ratio trade-off)
            efficiency_scores = []
            for r in successful_compression:
                ratio = r['compression_ratio']
                reduction = r['memory_reduction_percent']
                # Simple efficiency metric: reduction per unit compression
                efficiency = reduction / (1 - ratio) if ratio < 1 else 0
                efficiency_scores.append(efficiency)
            
            axes[2].bar(range(len(efficiency_scores)), efficiency_scores, alpha=0.7)
            axes[2].set_xlabel('Configuration Index')
            axes[2].set_ylabel('Efficiency Score')
            axes[2].set_title('Compression Efficiency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'compression_memory_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # Operations profiling plots
    if 'operations_analysis' in results:
        ops_data = results['operations_analysis']
        
        if ops_data:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Operations Memory Profiling', fontsize=16)
            
            operations = list(ops_data.keys())
            peak_memories = [ops_data[op]['peak_memory_mb'] for op in operations]
            memory_growths = [ops_data[op]['memory_growth'] for op in operations]
            
            # Peak memory by operation
            axes[0].bar(operations, peak_memories, alpha=0.7)
            axes[0].set_ylabel('Peak Memory (MB)')
            axes[0].set_title('Peak Memory by Operation')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Memory growth by operation
            axes[1].bar(operations, memory_growths, alpha=0.7)
            axes[1].set_ylabel('Memory Growth (MB)')
            axes[1].set_title('Memory Growth by Operation')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'operations_memory_profiling.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def save_results(results: Dict[str, Any], output_dir: str, filename: str = 'memory_benchmark_results.json'):
    """Save benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {filepath}")

def main():
    """Main memory benchmarking function."""
    parser = argparse.ArgumentParser(description='Memory benchmarking for ChunkedDecomp models')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='gpt2',
                       help='Model name for benchmarking')
    parser.add_argument('--config', type=str, 
                       help='Configuration file path')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    # Benchmark arguments
    parser.add_argument('--output_dir', type=str, default='memory_results',
                       help='Output directory for results')
    parser.add_argument('--scaling_analysis', action='store_true',
                       help='Run memory scaling analysis')
    parser.add_argument('--compression_analysis', action='store_true',
                       help='Run compression memory analysis')
    parser.add_argument('--detailed_profiling', action='store_true',
                       help='Run detailed memory profiling')
    
    # Scaling parameters
    parser.add_argument('--batch_sizes', nargs='+', type=int, 
                       default=[1, 2, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--sequence_lengths', nargs='+', type=int,
                       default=[128, 256, 512, 1024],
                       help='Sequence lengths to test')
    parser.add_argument('--max_batch_size', type=int, default=16,
                       help='Maximum batch size for scaling')
    parser.add_argument('--max_seq_length', type=int, default=1024,
                       help='Maximum sequence length for scaling')
    
    # Options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {
        'model_name': args.model_name,
        'device': args.device,
        'timestamp': time.time(),
        'config': config
    }
    
    try:
        # Memory scaling analysis
        if args.scaling_analysis:
            logger.info("Running memory scaling analysis...")
            
            scaling_results = memory_scaling_analysis(
                model_name=args.model_name,
                batch_sizes=args.batch_sizes,
                sequence_lengths=args.sequence_lengths,
                device=args.device,
                output_dir=args.output_dir
            )
            
            all_results['scaling_analysis'] = scaling_results
            
            # Save intermediate results
            save_results(scaling_results, args.output_dir, 'scaling_analysis_results.json')
        
        # Compression analysis
        if args.compression_analysis:
            logger.info("Running compression memory analysis...")
            
            # Create test compression configs
            compression_configs = [
                {'compression_ratio': 0.5, 'chunk_size': 32},
                {'compression_ratio': 0.3, 'chunk_size': 64},
                {'compression_ratio': 0.7, 'chunk_size': 128},
                {'compression_ratio': 0.4, 'chunk_size': 64},
                {'compression_ratio': 0.6, 'chunk_size': 32}
            ]
            
            if 'compression' in config:
                compression_configs = config['compression'].get('configs', compression_configs)
            
            compression_results = compression_memory_analysis(
                model_name=args.model_name,
                compression_configs=compression_configs,
                device=args.device,
                output_dir=args.output_dir
            )
            
            all_results['compression_analysis'] = compression_results
            
            # Save intermediate results
            save_results(compression_results, args.output_dir, 'compression_analysis_results.json')
        
        # Detailed profiling
        if args.detailed_profiling:
            logger.info("Running detailed memory profiling...")
            
            operations = ['forward_pass', 'generation', 'gradient_computation', 
                         'large_batch', 'long_sequence']
            
            profiling_results = detailed_memory_profiling(
                model_name=args.model_name,
                operations=operations,
                device=args.device,
                output_dir=args.output_dir
            )
            
            all_results['detailed_profiling'] = profiling_results
            
            # Save intermediate results
            save_results(profiling_results, args.output_dir, 'detailed_profiling_results.json')
        
        # Create visualizations
        logger.info("Creating visualization plots...")
        
        if args.scaling_analysis and 'scaling_analysis' in all_results:
            create_memory_plots(all_results['scaling_analysis'], args.output_dir)
        
        if args.compression_analysis and 'compression_analysis' in all_results:
            create_memory_plots(all_results['compression_analysis'], args.output_dir)
        
        if args.detailed_profiling and 'detailed_profiling' in all_results:
            create_memory_plots(all_results['detailed_profiling'], args.output_dir)
        
        # Save final results
        save_results(all_results, args.output_dir, 'complete_memory_benchmark.json')
        
        logger.info("Memory benchmarking completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("MEMORY BENCHMARK SUMMARY")
        print("="*60)
        print(f"Model: {args.model_name}")
        print(f"Device: {args.device}")
        print(f"Results saved to: {args.output_dir}")
        
        if args.scaling_analysis and 'scaling_analysis' in all_results:
            scaling = all_results['scaling_analysis']
            successful_runs = [p for p in scaling['scaling_data'] if p['success']]
            if successful_runs:
                max_memory = max(p['peak_memory_mb'] for p in successful_runs)
                print(f"Peak memory observed: {max_memory:.2f} MB")
        
        if args.compression_analysis and 'compression_analysis' in all_results:
            compression = all_results['compression_analysis']
            baseline_mem = compression.get('baseline_memory', {}).get('peak_memory_mb', 0)
            successful_comp = [r for r in compression['compression_results'] if r['success']]
            if successful_comp:
                best_reduction = max(r['memory_reduction_percent'] for r in successful_comp)
                print(f"Best memory reduction: {best_reduction:.1f}%")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        raise

if __name__ == "__main__":
    main()
