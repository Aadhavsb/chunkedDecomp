#!/usr/bin/env python3
"""
Model Evaluation Script for ChunkedDecomp

This script provides standalone model evaluation capabilities including:
- Performance metrics calculation
- Memory usage analysis
- Generation quality assessment
- Model comparison studies
- Benchmark evaluation

Usage:
    python evaluate_model.py --config configs/model_configs.yaml --model_path saved_models/compressed_model.pt
    python evaluate_model.py --model_name gpt2 --dataset wikitext-2 --compare_original
    python evaluate_model.py --batch_evaluation --output_dir results/
"""

import argparse
import os
import sys
import time
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.chunked_decomp import ChunkedDecomp
from models.compressed_model import CompressedModelWrapper
from evaluation.performance_evaluator import PerformanceEvaluator
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
            logging.FileHandler('evaluation.log')
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

def load_model(model_path: str, model_name: str = None, device: str = 'auto') -> torch.nn.Module:
    """Load model from path or create new model."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_path and os.path.exists(model_path):
        logging.info(f"Loading model from {model_path}")
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            # Load PyTorch model
            model = torch.load(model_path, map_location=device)
        else:
            # Load compressed model wrapper
            model = CompressedModelWrapper.load_model(model_path)
        model.to(device)
        return model
    elif model_name:
        logging.info(f"Creating new model: {model_name}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        return model
    else:
        raise ValueError("Either model_path or model_name must be provided")

def evaluate_single_model(
    model: torch.nn.Module,
    dataset_name: str,
    config: Dict[str, Any],
    output_dir: str,
    device: str = 'auto'
) -> Dict[str, Any]:
    """Evaluate a single model comprehensively."""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    # Setup data
    data_manager = DatasetManager()
    test_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=config.get('eval_batch_size', 8),
        max_length=config.get('max_length', 512)
    )
    
    # Performance evaluation
    logging.info("Running performance evaluation...")
    evaluator = PerformanceEvaluator(device=device)
    
    with MemoryTracker(device=device) as tracker:
        perf_results = evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            max_samples=config.get('max_eval_samples', 1000)
        )
    
    results['performance'] = perf_results
    results['memory_usage'] = tracker.get_stats()
    
    # Memory profiling
    logging.info("Running memory profiling...")
    profiler = MemoryProfiler(device=device)
    
    profiler.start_profiling()
    
    # Run inference with profiling
    model.eval()
    sample_batch = next(iter(test_loader))
    with torch.no_grad():
        profiler.mark_operation("inference_start")
        if hasattr(model, 'generate'):
            outputs = model.generate(
                sample_batch['input_ids'][:2].to(device),
                max_length=config.get('generation_max_length', 100),
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        else:
            outputs = model(sample_batch['input_ids'][:2].to(device))
        profiler.mark_operation("inference_end")
    
    memory_profile = profiler.stop_profiling()
    results['memory_profile'] = memory_profile
    
    # Benchmark evaluation
    logging.info("Running benchmark evaluation...")
    benchmark = ComprehensiveBenchmark(device=device)
    
    benchmark_results = benchmark.run_inference_benchmark(
        model=model,
        input_data=test_loader,
        batch_sizes=[1, 4, 8],
        sequence_lengths=[128, 256, 512],
        num_runs=config.get('benchmark_runs', 3)
    )
    
    results['benchmark'] = benchmark_results
    
    # Generate sample outputs
    logging.info("Generating sample outputs...")
    sample_outputs = generate_sample_outputs(
        model=model,
        data_manager=data_manager,
        dataset_name=dataset_name,
        device=device,
        num_samples=config.get('num_sample_outputs', 5)
    )
    
    results['sample_outputs'] = sample_outputs
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualizations
    create_evaluation_plots(results, output_dir)
    
    logging.info(f"Evaluation complete. Results saved to {output_dir}")
    return results

def generate_sample_outputs(
    model: torch.nn.Module,
    data_manager: DatasetManager,
    dataset_name: str,
    device: str,
    num_samples: int = 5
) -> List[Dict[str, str]]:
    """Generate sample outputs for qualitative evaluation."""
    
    test_data = data_manager.get_raw_dataset(dataset_name, split='test')
    sample_inputs = test_data[:num_samples] if hasattr(test_data, '__getitem__') else list(test_data)[:num_samples]
    
    tokenizer = data_manager.get_tokenizer(dataset_name)
    outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(sample_inputs):
            if isinstance(sample, dict):
                text = sample.get('text', str(sample))
            else:
                text = str(sample)
            
            # Truncate input for generation
            input_text = text[:100] + "..." if len(text) > 100 else text
            
            try:
                inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=50)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                if hasattr(model, 'generate'):
                    generated = model.generate(
                        inputs['input_ids'],
                        max_length=150,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                else:
                    # For models without generate method
                    output = model(**inputs)
                    generated_text = "Model output (logits only)"
                
                outputs.append({
                    'sample_id': i,
                    'input': input_text,
                    'output': generated_text,
                    'original_text': text
                })
                
            except Exception as e:
                logging.warning(f"Error generating sample {i}: {e}")
                outputs.append({
                    'sample_id': i,
                    'input': input_text,
                    'output': f"Generation failed: {str(e)}",
                    'original_text': text
                })
    
    return outputs

def compare_models(
    model_paths: List[str],
    model_names: List[str],
    dataset_name: str,
    config: Dict[str, Any],
    output_dir: str,
    device: str = 'auto'
) -> Dict[str, Any]:
    """Compare multiple models."""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models = []
    model_labels = []
    
    # Load models
    for i, (path, name) in enumerate(zip(model_paths, model_names)):
        try:
            model = load_model(path, name, device)
            models.append(model)
            model_labels.append(f"{name}_{i}" if name else f"model_{i}")
        except Exception as e:
            logging.error(f"Failed to load model {path}/{name}: {e}")
            continue
    
    if len(models) < 2:
        raise ValueError("Need at least 2 models for comparison")
    
    # Setup evaluation
    evaluator = PerformanceEvaluator(device=device)
    data_manager = DatasetManager()
    test_loader = data_manager.get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=config.get('eval_batch_size', 8),
        max_length=config.get('max_length', 512)
    )
    
    comparison_results = {}
    
    # Evaluate each model
    for i, (model, label) in enumerate(zip(models, model_labels)):
        logging.info(f"Evaluating model {i+1}/{len(models)}: {label}")
        
        with MemoryTracker(device=device) as tracker:
            perf_results = evaluator.evaluate_model(
                model=model,
                test_loader=test_loader,
                max_samples=config.get('max_eval_samples', 1000)
            )
        
        comparison_results[label] = {
            'performance': perf_results,
            'memory_usage': tracker.get_stats()
        }
    
    # Run comparative analysis
    logging.info("Running comparative analysis...")
    comparative_results = evaluator.compare_models(
        models=models,
        model_names=model_labels,
        test_loader=test_loader,
        max_samples=config.get('max_eval_samples', 1000)
    )
    
    comparison_results['comparative_analysis'] = comparative_results
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'model_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    # Create comparison plots
    create_comparison_plots(comparison_results, output_dir)
    
    logging.info(f"Model comparison complete. Results saved to {output_dir}")
    return comparison_results

def create_evaluation_plots(results: Dict[str, Any], output_dir: str):
    """Create evaluation visualization plots."""
    
    plt.style.use('seaborn-v0_8')
    
    # Performance metrics plot
    if 'performance' in results:
        perf = results['performance']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Evaluation', fontsize=16)
        
        # Perplexity over time
        if 'perplexity_history' in perf:
            axes[0, 0].plot(perf['perplexity_history'])
            axes[0, 0].set_title('Perplexity Over Batches')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Perplexity')
        
        # Memory usage
        if 'memory_profile' in results:
            mem_data = results['memory_profile']
            if 'timeline' in mem_data:
                timeline = mem_data['timeline']
                axes[0, 1].plot([t['timestamp'] for t in timeline], 
                               [t['memory_used'] for t in timeline])
                axes[0, 1].set_title('Memory Usage Timeline')
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('Memory (MB)')
        
        # Inference timing
        if 'benchmark' in results:
            bench = results['benchmark']
            if 'latency_results' in bench:
                latencies = bench['latency_results']
                axes[1, 0].hist(latencies, bins=20, alpha=0.7)
                axes[1, 0].set_title('Inference Latency Distribution')
                axes[1, 0].set_xlabel('Latency (ms)')
                axes[1, 0].set_ylabel('Frequency')
        
        # Throughput vs batch size
        if 'benchmark' in results and 'throughput_results' in results['benchmark']:
            throughput = results['benchmark']['throughput_results']
            if throughput:
                batch_sizes = list(throughput.keys())
                throughputs = list(throughput.values())
                axes[1, 1].plot(batch_sizes, throughputs, marker='o')
                axes[1, 1].set_title('Throughput vs Batch Size')
                axes[1, 1].set_xlabel('Batch Size')
                axes[1, 1].set_ylabel('Throughput (samples/sec)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_comparison_plots(results: Dict[str, Any], output_dir: str):
    """Create model comparison plots."""
    
    plt.style.use('seaborn-v0_8')
    
    # Extract comparison data
    model_names = []
    perplexities = []
    memory_usage = []
    inference_times = []
    
    for model_name, model_results in results.items():
        if model_name == 'comparative_analysis':
            continue
            
        model_names.append(model_name)
        
        if 'performance' in model_results:
            perf = model_results['performance']
            perplexities.append(perf.get('perplexity', 0))
        else:
            perplexities.append(0)
        
        if 'memory_usage' in model_results:
            mem = model_results['memory_usage']
            memory_usage.append(mem.get('peak_memory_mb', 0))
        else:
            memory_usage.append(0)
        
        inference_times.append(0)  # Placeholder
    
    if len(model_names) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Comparison', fontsize=16)
        
        # Perplexity comparison
        axes[0].bar(model_names, perplexities)
        axes[0].set_title('Perplexity Comparison')
        axes[0].set_ylabel('Perplexity')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        axes[1].bar(model_names, memory_usage)
        axes[1].set_title('Peak Memory Usage')
        axes[1].set_ylabel('Memory (MB)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Create a summary table
        comparison_data = pd.DataFrame({
            'Model': model_names,
            'Perplexity': perplexities,
            'Memory (MB)': memory_usage
        })
        
        axes[2].axis('tight')
        axes[2].axis('off')
        table = axes[2].table(cellText=comparison_data.values,
                             colLabels=comparison_data.columns,
                             cellLoc='center',
                             loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[2].set_title('Summary Table')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate ChunkedDecomp models')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, help='Path to saved model')
    parser.add_argument('--model_name', type=str, help='Model name (e.g., gpt2)')
    parser.add_argument('--config', type=str, default='configs/model_configs.yaml',
                       help='Configuration file path')
    
    # Evaluation arguments
    parser.add_argument('--dataset', type=str, default='wikitext-2',
                       help='Dataset for evaluation')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    # Comparison arguments
    parser.add_argument('--compare_models', nargs='+',
                       help='Paths to models for comparison')
    parser.add_argument('--compare_names', nargs='+',
                       help='Names for comparison models')
    parser.add_argument('--compare_original', action='store_true',
                       help='Compare with original uncompressed model')
    
    # Options
    parser.add_argument('--batch_evaluation', action='store_true',
                       help='Run batch evaluation mode')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration
    config = load_config(args.config)
    eval_config = config.get('evaluation', {})
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.compare_models or args.compare_original:
            # Model comparison mode
            model_paths = args.compare_models or []
            model_names = args.compare_names or []
            
            if args.compare_original and args.model_name:
                model_paths.append(None)
                model_names.append(args.model_name)
            
            if args.model_path:
                model_paths.append(args.model_path)
                model_names.append(args.model_name or 'compressed_model')
            
            if len(model_paths) < 2:
                logger.error("Need at least 2 models for comparison")
                return
            
            results = compare_models(
                model_paths=model_paths,
                model_names=model_names,
                dataset_name=args.dataset,
                config=eval_config,
                output_dir=args.output_dir,
                device=args.device
            )
            
        else:
            # Single model evaluation
            model = load_model(args.model_path, args.model_name, args.device)
            
            results = evaluate_single_model(
                model=model,
                dataset_name=args.dataset,
                config=eval_config,
                output_dir=args.output_dir,
                device=args.device
            )
        
        logger.info("Evaluation completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'performance' in results:
            perf = results['performance']
            print(f"Perplexity: {perf.get('perplexity', 'N/A'):.4f}")
            print(f"Loss: {perf.get('loss', 'N/A'):.4f}")
        
        if 'memory_usage' in results:
            mem = results['memory_usage']
            print(f"Peak Memory: {mem.get('peak_memory_mb', 'N/A'):.2f} MB")
        
        print(f"\nDetailed results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
