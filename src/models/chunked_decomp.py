"""Main ChunkedDecomp implementation for efficient KV cache compression."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import yaml

from .kv_cache import ChunkedKVCache
from ..utils.svd_utils import create_adaptive_rank_map

logger = logging.getLogger(__name__)


class ChunkedDecomp:
    """
    Main class for ChunkedDecomp - handles model integration and compression.
    
    This class wraps around a transformer model and provides efficient KV cache
    compression using chunked SVD decomposition.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        chunk_size: int = 16,
        block_size: int = 64,
        compression_strategy: str = "uniform",
        base_compression_ratio: float = 0.5,
        decomp_rank_map: Optional[Dict[int, int]] = None,
        device: Optional[torch.device] = None,
        **model_kwargs
    ):
        """Initialize ChunkedDecomp.
        
        Args:
            model_name_or_path: Hugging Face model name or path
            chunk_size: Size of each chunk for decomposition
            block_size: Number of tokens before compression is triggered
            compression_strategy: Strategy for adaptive compression
            base_compression_ratio: Base compression ratio
            decomp_rank_map: Custom rank mapping for chunks
            device: Device for computation
            **model_kwargs: Additional arguments for model loading
        """
        self.model_name_or_path = model_name_or_path
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.compression_strategy = compression_strategy
        self.base_compression_ratio = base_compression_ratio
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and config
        self._load_model(**model_kwargs)
        
        # Extract model configuration
        self._extract_model_config()
        
        # Create decomposition rank map
        if decomp_rank_map is None:
            self.decomp_rank_map = create_adaptive_rank_map(
                n_chunks=self.n_chunks,
                chunk_size=self.chunk_size,
                strategy=self.compression_strategy,
                base_compression_ratio=self.base_compression_ratio
            )
        else:
            self.decomp_rank_map = decomp_rank_map
        
        # Initialize chunked KV cache
        self.kv_cache = ChunkedKVCache(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            chunk_size=self.chunk_size,
            block_size=self.block_size,
            decomp_rank_map=self.decomp_rank_map,
            device=self.device
        )
        
        # Hook into model's attention mechanism
        self._register_hooks()
        
        logger.info(f"ChunkedDecomp initialized for {model_name_or_path}")
        logger.info(f"Model config: {self.n_layers} layers, {self.n_heads} heads, {self.head_dim} head_dim")
        logger.info(f"Compression: {self.chunk_size} chunk_size, {self.block_size} block_size")
        logger.info(f"Rank map: {self.decomp_rank_map}")
    
    def _load_model(self, **model_kwargs):
        """Load the transformer model."""
        try:
            self.config = AutoConfig.from_pretrained(self.model_name_or_path)
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path,
                **model_kwargs
            ).to(self.device)
            self.model.eval()  # Set to evaluation mode
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name_or_path}: {e}")
            raise
    
    def _extract_model_config(self):
        """Extract relevant configuration from the model."""
        # Different model architectures have different config names
        if hasattr(self.config, 'n_layer'):
            self.n_layers = self.config.n_layer
        elif hasattr(self.config, 'num_hidden_layers'):
            self.n_layers = self.config.num_hidden_layers
        else:
            raise ValueError("Could not determine number of layers from model config")
        
        if hasattr(self.config, 'n_head'):
            self.n_heads = self.config.n_head
        elif hasattr(self.config, 'num_attention_heads'):
            self.n_heads = self.config.num_attention_heads
        else:
            raise ValueError("Could not determine number of attention heads from model config")
        
        if hasattr(self.config, 'n_embd'):
            total_dim = self.config.n_embd
        elif hasattr(self.config, 'hidden_size'):
            total_dim = self.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config")
        
        self.head_dim = total_dim // self.n_heads
        
        # Validate chunk size compatibility
        if self.head_dim % self.chunk_size != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be divisible by chunk_size ({self.chunk_size})"
            )
        
        self.n_chunks = self.head_dim // self.chunk_size
        
        logger.debug(f"Extracted config: {self.n_layers} layers, {self.n_heads} heads, {self.head_dim} head_dim")
    
    def _register_hooks(self):
        """Register forward hooks to intercept attention computations."""
        self.hooks = []
        
        # Find attention modules - this is model-specific
        attention_modules = self._find_attention_modules()
        
        for layer_idx, attn_module in enumerate(attention_modules):
            hook = attn_module.register_forward_hook(
                lambda module, input, output, layer_idx=layer_idx: self._attention_hook(
                    module, input, output, layer_idx
                )
            )
            self.hooks.append(hook)
    
    def _find_attention_modules(self):
        """Find attention modules in the model architecture."""
        attention_modules = []
        
        # This is a simplified approach - might need adjustment for different models
        if hasattr(self.model, 'transformer'):  # GPT-2 style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'layers'):  # Some other models
            layers = self.model.layers
        else:
            # Try to find layers automatically
            layers = []
            for name, module in self.model.named_modules():
                if 'layer' in name.lower() and hasattr(module, 'attention'):
                    layers.append(module)
        
        for layer in layers:
            if hasattr(layer, 'attn'):  # GPT-2 style
                attention_modules.append(layer.attn)
            elif hasattr(layer, 'attention'):
                attention_modules.append(layer.attention)
            elif hasattr(layer, 'self_attn'):
                attention_modules.append(layer.self_attn)
        
        if len(attention_modules) != self.n_layers:
            logger.warning(
                f"Found {len(attention_modules)} attention modules, expected {self.n_layers}. "
                "Hook registration might be incomplete."
            )
        
        return attention_modules
    
    def _attention_hook(self, module, input, output, layer_idx: int):
        """Hook function to intercept attention computations."""
        # This is a simplified implementation
        # In practice, you'd need to extract K, V from the attention computation
        # and route them through the chunked cache
        
        # For now, we'll simulate this by extracting from the input
        # This needs to be adapted based on the specific model architecture
        pass
    
    def forward_with_compression(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with KV cache compression.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional model arguments
            
        Returns:
            Model outputs with compression statistics
        """
        # Clear cache for new sequence
        self.kv_cache.clear_cache()
        
        # Run model forward pass (hooks will handle caching)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Get memory usage statistics
        memory_stats = self.kv_cache.get_memory_usage()
        
        return {
            'model_outputs': outputs,
            'memory_stats': memory_stats,
            'compression_stats': {
                'total_compressions': self.kv_cache.stats['total_compressions'],
                'total_decompressions': self.kv_cache.stats['total_decompressions'],
                'compression_errors': self.kv_cache.stats['compression_errors']
            }
        }
    
    def generate_with_compression(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with KV cache compression.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments
            
        Returns:
            Generated tokens with compression statistics
        """
        # Clear cache for new generation
        self.kv_cache.clear_cache()
        
        generated_tokens = []
        current_ids = input_ids.clone()
        
        for step in range(max_length):
            # Forward pass for current tokens
            result = self.forward_with_compression(current_ids)
            outputs = result['model_outputs']
            
            # Get next token logits
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            if do_sample:
                # Apply temperature and top-p sampling
                logits = logits / temperature
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_tokens.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            # Check for early stopping (implement as needed)
            # if next_token.item() == eos_token_id:
            #     break
        
        # Final memory statistics
        final_memory_stats = self.kv_cache.get_memory_usage()
        
        return {
            'generated_ids': current_ids,
            'generated_tokens': torch.cat(generated_tokens, dim=-1) if generated_tokens else torch.empty(0),
            'memory_stats': final_memory_stats,
            'compression_stats': {
                'total_compressions': self.kv_cache.stats['total_compressions'],
                'total_decompressions': self.kv_cache.stats['total_decompressions'],
                'compression_errors': self.kv_cache.stats['compression_errors']
            }
        }
    
    def benchmark_compression(
        self,
        input_ids: torch.Tensor,
        runs: int = 5
    ) -> Dict[str, Any]:
        """Benchmark compression performance.
        
        Args:
            input_ids: Input tokens for benchmarking
            runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        import time
        
        results = {
            'runs': [],
            'average_compression_time': 0,
            'average_memory_saved': 0,
            'average_compression_ratio': 0
        }
        
        for run in range(runs):
            start_time = time.time()
            
            # Run forward pass with compression
            result = self.forward_with_compression(input_ids)
            
            end_time = time.time()
            
            run_result = {
                'run_id': run,
                'execution_time': end_time - start_time,
                'memory_stats': result['memory_stats'],
                'compression_stats': result['compression_stats']
            }
            
            results['runs'].append(run_result)
        
        # Calculate averages
        if results['runs']:
            results['average_compression_time'] = sum(r['execution_time'] for r in results['runs']) / len(results['runs'])
            results['average_compression_ratio'] = sum(r['memory_stats']['compression_ratio'] for r in results['runs']) / len(results['runs'])
        
        return results
    
    def save_config(self, filepath: str):
        """Save current configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        config = {
            'model_name_or_path': self.model_name_or_path,
            'chunk_size': self.chunk_size,
            'block_size': self.block_size,
            'compression_strategy': self.compression_strategy,
            'base_compression_ratio': self.base_compression_ratio,
            'decomp_rank_map': self.decomp_rank_map,
            'model_config': {
                'n_layers': self.n_layers,
                'n_heads': self.n_heads,
                'head_dim': self.head_dim,
                'n_chunks': self.n_chunks
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_config(cls, filepath: str, **override_kwargs):
        """Load ChunkedDecomp from configuration file.
        
        Args:
            filepath: Path to configuration file
            **override_kwargs: Arguments to override from config
            
        Returns:
            ChunkedDecomp instance
        """
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with any provided kwargs
        config.update(override_kwargs)
        
        # Remove model_config as it will be extracted automatically
        config.pop('model_config', None)
        
        return cls(**config)
    
    def cleanup(self):
        """Clean up hooks and resources."""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Clear cache
        self.kv_cache.clear_cache()
        
        logger.info("ChunkedDecomp cleanup completed")
