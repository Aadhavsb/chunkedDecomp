"""Performance evaluation utilities for chunked decomposition models."""

import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    accuracy: float
    perplexity: float
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    inference_time: float = 0.0
    memory_usage: Dict[str, float] = None
    throughput: float = 0.0  # tokens/second
    compression_ratio: Optional[float] = None
    reconstruction_error: Optional[float] = None


class PerformanceEvaluator:
    """Evaluates performance of compressed models."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize performance evaluator.
        
        Args:
            device: Device to run evaluation on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_history = []
        
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        tokenizer: Any = None,
        max_batches: Optional[int] = None,
        compute_perplexity: bool = True,
        compute_generation_metrics: bool = False
    ) -> PerformanceMetrics:
        """Evaluate model performance on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            tokenizer: Tokenizer for text generation metrics
            max_batches: Maximum number of batches to evaluate
            compute_perplexity: Whether to compute perplexity
            compute_generation_metrics: Whether to compute generation metrics
            
        Returns:
            Performance metrics
        """
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Time inference
                start_time = time.time()
                
                if isinstance(batch, dict):
                    outputs = model(**batch)
                else:
                    outputs = model(batch)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Extract loss and logits
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                    total_loss += loss.item()
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    
                    # Calculate accuracy for classification tasks
                    if 'labels' in batch:
                        labels = batch['labels']
                        predictions = torch.argmax(logits, dim=-1)
                        
                        # Handle sequence labeling
                        if labels.dim() > 1:
                            mask = labels != -100  # Ignore padding tokens
                            correct_predictions += (predictions == labels)[mask].sum().item()
                            total_predictions += mask.sum().item()
                            total_tokens += mask.sum().item()
                        else:
                            correct_predictions += (predictions == labels).sum().item()
                            total_predictions += labels.size(0)
                            total_tokens += labels.size(0)
        
        # Calculate metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if compute_perplexity else 0.0
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        throughput = total_tokens / sum(inference_times) if inference_times else 0.0
        
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            perplexity=perplexity,
            inference_time=avg_inference_time,
            throughput=throughput
        )
        
        # Add generation metrics if requested
        if compute_generation_metrics and tokenizer is not None:
            gen_metrics = self._compute_generation_metrics(model, dataloader, tokenizer)
            metrics.bleu_score = gen_metrics.get('bleu_score')
            metrics.rouge_scores = gen_metrics.get('rouge_scores')
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _compute_generation_metrics(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        tokenizer: Any,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Compute text generation metrics like BLEU and ROUGE.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader with evaluation data
            tokenizer: Tokenizer for text processing
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Generation metrics
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning("NLTK or rouge-score not available. Skipping generation metrics.")
            return {}
        
        model.eval()
        bleu_scores = []
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_samples:
                    break
                
                if 'input_ids' not in batch or 'labels' not in batch:
                    continue
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Generate text
                generated = model.generate(
                    input_ids,
                    max_length=input_ids.size(1) + 50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # Decode texts
                generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                reference_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # Calculate BLEU and ROUGE for each sample
                for gen_text, ref_text in zip(generated_texts, reference_texts):
                    # BLEU score
                    gen_tokens = gen_text.split()
                    ref_tokens = [ref_text.split()]
                    bleu = sentence_bleu(ref_tokens, gen_tokens)
                    bleu_scores.append(bleu)
                    
                    # ROUGE scores
                    scores = scorer.score(ref_text, gen_text)
                    rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                    rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                    rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'bleu_score': np.mean(bleu_scores) if bleu_scores else 0.0,
            'rouge_scores': {k: np.mean(v) for k, v in rouge_scores.items()}
        }
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        dataloader: torch.utils.data.DataLoader,
        **eval_kwargs
    ) -> Dict[str, PerformanceMetrics]:
        """Compare performance of multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            dataloader: DataLoader with evaluation data
            **eval_kwargs: Additional arguments for evaluate_model
            
        Returns:
            Dictionary of model_name -> metrics
        """
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            metrics = self.evaluate_model(model, dataloader, **eval_kwargs)
            results[model_name] = metrics
            
        return results
    
    def benchmark_compression_ratio(
        self,
        original_model: torch.nn.Module,
        compressed_model: torch.nn.Module
    ) -> float:
        """Calculate compression ratio between original and compressed models.
        
        Args:
            original_model: Original uncompressed model
            compressed_model: Compressed model
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        def get_model_size(model):
            """Get model size in bytes."""
            param_size = 0
            for param in model.parameters():
                param_size += param.numel() * param.element_size()
            return param_size
        
        original_size = get_model_size(original_model)
        compressed_size = get_model_size(compressed_model)
        
        return original_size / compressed_size if compressed_size > 0 else 0.0
    
    def evaluate_reconstruction_error(
        self,
        original_outputs: torch.Tensor,
        reconstructed_outputs: torch.Tensor,
        metric: str = 'mse'
    ) -> float:
        """Evaluate reconstruction error between original and reconstructed outputs.
        
        Args:
            original_outputs: Original model outputs
            reconstructed_outputs: Reconstructed model outputs
            metric: Error metric ('mse', 'mae', 'cosine')
            
        Returns:
            Reconstruction error
        """
        if metric == 'mse':
            error = F.mse_loss(reconstructed_outputs, original_outputs).item()
        elif metric == 'mae':
            error = F.l1_loss(reconstructed_outputs, original_outputs).item()
        elif metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            similarity = F.cosine_similarity(
                original_outputs.flatten(),
                reconstructed_outputs.flatten(),
                dim=0
            )
            error = 1.0 - similarity.item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return error
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, PerformanceMetrics],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of metrics across models.
        
        Args:
            metrics_dict: Dictionary of model_name -> metrics
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(metrics_dict.keys())
        
        # Accuracy comparison
        accuracies = [metrics_dict[model].accuracy for model in models]
        axes[0, 0].bar(models, accuracies)
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Perplexity comparison
        perplexities = [metrics_dict[model].perplexity for model in models]
        axes[0, 1].bar(models, perplexities)
        axes[0, 1].set_title('Perplexity (Lower is Better)')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        inference_times = [metrics_dict[model].inference_time for model in models]
        axes[1, 0].bar(models, inference_times)
        axes[1, 0].set_title('Inference Time (Lower is Better)')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        throughputs = [metrics_dict[model].throughput for model in models]
        axes[1, 1].bar(models, throughputs)
        axes[1, 1].set_title('Throughput (Higher is Better)')
        axes[1, 1].set_ylabel('Tokens/second')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def export_metrics(
        self,
        metrics: PerformanceMetrics,
        filepath: str,
        format: str = 'json'
    ):
        """Export metrics to file.
        
        Args:
            metrics: Metrics to export
            filepath: Output file path
            format: Export format ('json', 'csv')
        """
        from dataclasses import asdict
        import json
        import csv
        
        metrics_dict = asdict(metrics)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        elif format == 'csv':
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in metrics_dict.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            writer.writerow([f"{key}_{subkey}", subvalue])
                    else:
                        writer.writerow([key, value])
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Metrics exported to {filepath}")


@contextmanager
def performance_profiler(model: torch.nn.Module, device: Optional[torch.device] = None):
    """Context manager for performance profiling.
    
    Args:
        model: Model to profile
        device: Device to run profiling on
        
    Usage:
        with performance_profiler(model) as profiler:
            outputs = model(inputs)
        print(profiler.get_summary())
    """
    profiler = PerformanceProfiler(model, device)
    profiler.start()
    
    try:
        yield profiler
    finally:
        profiler.stop()


class PerformanceProfiler:
    """Detailed performance profiler for models."""
    
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """Initialize performance profiler.
        
        Args:
            model: Model to profile
            device: Device to run profiling on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = None
        self.results = None
        
    def start(self):
        """Start profiling."""
        if torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()
    
    def stop(self):
        """Stop profiling."""
        if self.profiler:
            self.profiler.stop()
            self.results = self.profiler
    
    def get_summary(self) -> str:
        """Get profiling summary.
        
        Returns:
            Profiling summary string
        """
        if not self.results:
            return "No profiling results available"
        
        return self.results.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        )
    
    def export_trace(self, filepath: str):
        """Export profiling trace.
        
        Args:
            filepath: Path to save trace file
        """
        if self.results:
            self.results.export_chrome_trace(filepath)
            logger.info(f"Profiling trace exported to {filepath}")


def compare_model_efficiency(
    models: Dict[str, torch.nn.Module],
    input_data: torch.Tensor,
    num_runs: int = 10
) -> Dict[str, Dict[str, float]]:
    """Compare efficiency metrics across multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        input_data: Input tensor for inference
        num_runs: Number of inference runs for averaging
        
    Returns:
        Dictionary of model_name -> efficiency metrics
    """
    results = {}
    
    for model_name, model in models.items():
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)
        
        # Timed runs
        inference_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                outputs = model(input_data)
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        # Calculate metrics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        results[model_name] = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput': input_data.size(1) / avg_time if avg_time > 0 else 0  # tokens/second
        }
    
    return results
