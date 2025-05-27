"""Compressed model wrapper for chunked decomposition models."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from transformers import PreTrainedModel, AutoConfig, AutoModel
import warnings

from .chunked_decomp import ChunkedDecomp
from ..utils.svd_utils import SVDCompressor
from ..utils.memory_utils import get_model_memory_footprint

logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics about model compression."""
    original_parameters: int
    compressed_parameters: int
    compression_ratio: float
    memory_saved_mb: float
    compression_time: float
    reconstruction_error: float


class CompressedModelWrapper(nn.Module):
    """Wrapper for compressed transformer models with chunked decomposition."""
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        compression_config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize compressed model wrapper.
        
        Args:
            base_model: Original transformer model to compress
            compression_config: Configuration for compression
            device: Device to place model on
        """
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compression_config = compression_config
        
        # Store original model info
        self.model_config = base_model.config
        self.model_name = getattr(base_model, 'name_or_path', 'unknown')
        
        # Initialize chunked decomposition
        self.chunked_decomp = ChunkedDecomp(
            base_model=base_model,
            config=compression_config,
            device=self.device
        )
        
        # Apply compression
        self.compression_stats = self._apply_compression()
        
        # Store the compressed model
        self.model = self.chunked_decomp.model
        
        logger.info(f"Compressed model initialized with {self.compression_stats.compression_ratio:.2f}x compression")
    
    def _apply_compression(self) -> CompressionStats:
        """Apply compression to the model and collect statistics.
        
        Returns:
            Compression statistics
        """
        import time
        
        # Get original model stats
        original_footprint = get_model_memory_footprint(self.chunked_decomp.model)
        original_params = original_footprint['parameter_count']
        
        start_time = time.time()
        
        # Apply compression
        compression_result = self.chunked_decomp.compress_model(
            target_compression_ratio=self.compression_config.get('target_ratio', 0.5),
            chunk_size=self.compression_config.get('chunk_size', 128),
            adaptive_rank=self.compression_config.get('adaptive_rank', True)
        )
        
        compression_time = time.time() - start_time
        
        # Get compressed model stats
        compressed_footprint = get_model_memory_footprint(self.chunked_decomp.model)
        compressed_params = compressed_footprint['parameter_count']
        
        # Calculate statistics
        compression_ratio = original_params / compressed_params if compressed_params > 0 else 1.0
        memory_saved = original_footprint['total_mb'] - compressed_footprint['total_mb']
        
        stats = CompressionStats(
            original_parameters=original_params,
            compressed_parameters=compressed_params,
            compression_ratio=compression_ratio,
            memory_saved_mb=memory_saved,
            compression_time=compression_time,
            reconstruction_error=compression_result.get('reconstruction_error', 0.0)
        )
        
        return stats
    
    def forward(self, *args, **kwargs):
        """Forward pass through the compressed model."""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generate text using the compressed model."""
        if hasattr(self.model, 'generate'):
            return self.model.generate(*args, **kwargs)
        else:
            raise NotImplementedError("Base model does not support generation")
    
    def get_compression_info(self) -> Dict[str, Any]:
        """Get detailed compression information.
        
        Returns:
            Dictionary with compression details
        """
        return {
            'model_name': self.model_name,
            'compression_config': self.compression_config,
            'compression_stats': {
                'original_parameters': self.compression_stats.original_parameters,
                'compressed_parameters': self.compression_stats.compressed_parameters,
                'compression_ratio': self.compression_stats.compression_ratio,
                'memory_saved_mb': self.compression_stats.memory_saved_mb,
                'compression_time': self.compression_stats.compression_time,
                'reconstruction_error': self.compression_stats.reconstruction_error
            },
            'chunked_decomp_stats': self.chunked_decomp.get_compression_stats(),
            'device': str(self.device)
        }
    
    def benchmark_performance(
        self,
        input_data: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark performance of the compressed model.
        
        Args:
            input_data: Input tensor for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Performance metrics
        """
        return self.chunked_decomp.benchmark_model(input_data, num_runs)
    
    def save_compressed_model(self, save_path: str):
        """Save the compressed model to disk.
        
        Args:
            save_path: Path to save the compressed model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = save_path / 'compressed_model.pt'
        torch.save(self.model.state_dict(), model_path)
        
        # Save compression info
        info_path = save_path / 'compression_info.json'
        with open(info_path, 'w') as f:
            json.dump(self.get_compression_info(), f, indent=2, default=str)
        
        # Save model config
        config_path = save_path / 'model_config.json'
        self.model_config.save_pretrained(save_path)
        
        # Save chunked decomp state
        chunked_decomp_path = save_path / 'chunked_decomp.pt'
        self.chunked_decomp.save_state(str(chunked_decomp_path))
        
        logger.info(f"Compressed model saved to {save_path}")
    
    @classmethod
    def load_compressed_model(
        cls,
        load_path: str,
        device: Optional[torch.device] = None
    ) -> 'CompressedModelWrapper':
        """Load a compressed model from disk.
        
        Args:
            load_path: Path to load the compressed model from
            device: Device to place model on
            
        Returns:
            Loaded compressed model wrapper
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {load_path}")
        
        # Load compression info
        info_path = load_path / 'compression_info.json'
        with open(info_path, 'r') as f:
            compression_info = json.load(f)
        
        # Load model config
        model_config = AutoConfig.from_pretrained(load_path)
        
        # Create base model (we'll replace its state)
        base_model = AutoModel.from_config(model_config)
        
        # Create wrapper
        wrapper = cls(
            base_model=base_model,
            compression_config=compression_info['compression_config'],
            device=device
        )
        
        # Load model state
        model_path = load_path / 'compressed_model.pt'
        state_dict = torch.load(model_path, map_location=device)
        wrapper.model.load_state_dict(state_dict)
        
        # Load chunked decomp state
        chunked_decomp_path = load_path / 'chunked_decomp.pt'
        if chunked_decomp_path.exists():
            wrapper.chunked_decomp.load_state(str(chunked_decomp_path))
        
        logger.info(f"Compressed model loaded from {load_path}")
        return wrapper
    
    def compare_with_original(
        self,
        original_model: PreTrainedModel,
        test_input: torch.Tensor,
        compute_metrics: bool = True
    ) -> Dict[str, Any]:
        """Compare compressed model with original model.
        
        Args:
            original_model: Original uncompressed model
            test_input: Test input for comparison
            compute_metrics: Whether to compute detailed metrics
            
        Returns:
            Comparison results
        """
        comparison = {
            'compression_ratio': self.compression_stats.compression_ratio,
            'memory_saved_mb': self.compression_stats.memory_saved_mb,
            'reconstruction_error': self.compression_stats.reconstruction_error
        }
        
        if compute_metrics:
            # Compare outputs
            original_model.eval()
            self.model.eval()
            
            with torch.no_grad():
                original_output = original_model(test_input)
                compressed_output = self.model(test_input)
                
                # Calculate output difference
                if hasattr(original_output, 'logits') and hasattr(compressed_output, 'logits'):
                    output_diff = torch.nn.functional.mse_loss(
                        compressed_output.logits,
                        original_output.logits
                    ).item()
                    comparison['output_mse'] = output_diff
                    
                    # Calculate cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        original_output.logits.flatten(),
                        compressed_output.logits.flatten(),
                        dim=0
                    ).item()
                    comparison['output_cosine_similarity'] = cos_sim
            
            # Compare model sizes
            original_footprint = get_model_memory_footprint(original_model)
            compressed_footprint = get_model_memory_footprint(self.model)
            
            comparison['model_sizes'] = {
                'original_mb': original_footprint['total_mb'],
                'compressed_mb': compressed_footprint['total_mb'],
                'size_reduction_mb': original_footprint['total_mb'] - compressed_footprint['total_mb'],
                'size_reduction_percent': (
                    (original_footprint['total_mb'] - compressed_footprint['total_mb']) / 
                    original_footprint['total_mb'] * 100
                ) if original_footprint['total_mb'] > 0 else 0
            }
            
            # Compare parameter counts
            comparison['parameter_counts'] = {
                'original': original_footprint['parameter_count'],
                'compressed': compressed_footprint['parameter_count'],
                'reduction': original_footprint['parameter_count'] - compressed_footprint['parameter_count'],
                'reduction_percent': (
                    (original_footprint['parameter_count'] - compressed_footprint['parameter_count']) /
                    original_footprint['parameter_count'] * 100
                ) if original_footprint['parameter_count'] > 0 else 0
            }
        
        return comparison
    
    def get_layer_compression_details(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed compression information for each layer.
        
        Returns:
            Dictionary with per-layer compression details
        """
        return self.chunked_decomp.get_layer_compression_stats()
    
    def adjust_compression(
        self,
        new_target_ratio: float,
        layers_to_adjust: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Dynamically adjust compression ratio.
        
        Args:
            new_target_ratio: New target compression ratio
            layers_to_adjust: Specific layers to adjust (if None, adjust all)
            
        Returns:
            Results of compression adjustment
        """
        logger.info(f"Adjusting compression ratio to {new_target_ratio}")
        
        # Update compression config
        self.compression_config['target_ratio'] = new_target_ratio
        
        # Apply new compression
        result = self.chunked_decomp.compress_model(
            target_compression_ratio=new_target_ratio,
            layers_to_compress=layers_to_adjust
        )
        
        # Update compression stats
        self.compression_stats = self._apply_compression()
        
        return {
            'new_compression_ratio': self.compression_stats.compression_ratio,
            'compression_result': result,
            'updated_stats': self.compression_stats
        }
    
    def enable_dynamic_compression(self, enable: bool = True):
        """Enable or disable dynamic compression during inference.
        
        Args:
            enable: Whether to enable dynamic compression
        """
        if hasattr(self.chunked_decomp, 'enable_dynamic_compression'):
            self.chunked_decomp.enable_dynamic_compression(enable)
            logger.info(f"Dynamic compression {'enabled' if enable else 'disabled'}")
        else:
            logger.warning("Dynamic compression not supported by this model")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of the compressed model.
        
        Returns:
            Memory usage statistics
        """
        footprint = get_model_memory_footprint(self.model)
        
        # Add KV cache memory if available
        kv_cache_memory = 0.0
        if hasattr(self.chunked_decomp, 'kv_cache') and self.chunked_decomp.kv_cache:
            kv_cache_stats = self.chunked_decomp.kv_cache.get_cache_stats()
            kv_cache_memory = kv_cache_stats.get('total_memory_mb', 0.0)
        
        return {
            'model_memory_mb': footprint['total_mb'],
            'kv_cache_memory_mb': kv_cache_memory,
            'total_memory_mb': footprint['total_mb'] + kv_cache_memory,
            'parameter_count': footprint['parameter_count']
        }
    
    def export_onnx(self, output_path: str, example_input: torch.Tensor):
        """Export compressed model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            example_input: Example input for tracing
        """
        try:
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                },
                opset_version=11
            )
            logger.info(f"ONNX model exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation of the compressed model."""
        return (
            f"CompressedModelWrapper(\n"
            f"  model_name='{self.model_name}',\n"
            f"  compression_ratio={self.compression_stats.compression_ratio:.2f}x,\n"
            f"  memory_saved={self.compression_stats.memory_saved_mb:.1f}MB,\n"
            f"  reconstruction_error={self.compression_stats.reconstruction_error:.6f},\n"
            f"  device='{self.device}'\n"
            f")"
        )


def create_compressed_model(
    model_name_or_path: str,
    compression_config: Dict[str, Any],
    device: Optional[torch.device] = None,
    model_kwargs: Optional[Dict[str, Any]] = None
) -> CompressedModelWrapper:
    """Create a compressed model from a model name or path.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        compression_config: Configuration for compression
        device: Device to place model on
        model_kwargs: Additional arguments for model loading
        
    Returns:
        Compressed model wrapper
    """
    from transformers import AutoModel
    
    model_kwargs = model_kwargs or {}
    
    # Load base model
    logger.info(f"Loading base model: {model_name_or_path}")
    base_model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    
    # Create compressed wrapper
    compressed_model = CompressedModelWrapper(
        base_model=base_model,
        compression_config=compression_config,
        device=device
    )
    
    return compressed_model


def compare_compression_configs(
    model_name_or_path: str,
    compression_configs: Dict[str, Dict[str, Any]],
    test_input: torch.Tensor,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, Any]]:
    """Compare different compression configurations on the same model.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        compression_configs: Dictionary of config_name -> compression_config
        test_input: Test input for comparison
        device: Device to place models on
        
    Returns:
        Dictionary of config_name -> comparison results
    """
    from transformers import AutoModel
    
    # Load original model once
    original_model = AutoModel.from_pretrained(model_name_or_path)
    if device:
        original_model = original_model.to(device)
    
    results = {}
    
    for config_name, config in compression_configs.items():
        logger.info(f"Testing compression config: {config_name}")
        
        try:
            # Create compressed model
            compressed_model = create_compressed_model(
                model_name_or_path,
                config,
                device
            )
            
            # Compare with original
            comparison = compressed_model.compare_with_original(
                original_model,
                test_input
            )
            
            # Add config info
            comparison['config'] = config
            comparison['config_name'] = config_name
            
            results[config_name] = comparison
            
        except Exception as e:
            logger.error(f"Failed to test config {config_name}: {e}")
            results[config_name] = {'error': str(e)}
    
    return results
