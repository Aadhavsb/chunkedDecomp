"""KV Cache implementation with chunked compression."""

import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

from ..utils.svd_utils import SVDCompressor, create_adaptive_rank_map

logger = logging.getLogger(__name__)


class ChunkedKVCache:
    """
    Manages KV cache with chunked compression using SVD.
    
    The cache operates in two phases:
    1. Standard cache: Holds recent tokens in full resolution
    2. Decomposed cache: Stores older tokens in compressed form
    """
    
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        chunk_size: int = 16,
        block_size: int = 64,
        decomp_rank_map: Optional[Dict[int, int]] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize chunked KV cache.
        
        Args:
            n_layers: Number of transformer layers
            n_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            chunk_size: Size of each chunk (head_dim must be divisible by this)
            block_size: Number of tokens before compression is triggered
            decomp_rank_map: Custom rank mapping for chunks
            device: Device for computations
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate chunk size
        if head_dim % chunk_size != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by chunk_size ({chunk_size})")
        
        self.n_chunks = head_dim // chunk_size
        
        # Create decomposition rank map
        self.decomp_rank_map = decomp_rank_map or {i: self.chunk_size // 2 for i in range(self.n_chunks)}
        
        # Initialize SVD compressor
        self.compressor = SVDCompressor(device=self.device)
        
        # Initialize cache structures
        self._initialize_cache()
        
        # Statistics
        self.stats = {
            'total_tokens_processed': 0,
            'total_compressions': 0,
            'total_decompressions': 0,
            'compression_errors': []
        }
    
    def _initialize_cache(self):
        """Initialize the dual cache structure."""
        # Standard cache: holds recent tokens
        # Structure: [layer][head] = List[Tuple[key_tensor, value_tensor]]
        self.standard_cache = defaultdict(lambda: defaultdict(list))
        
        # Decomposed cache: holds compressed older tokens
        # Structure: [layer][head][chunk_idx] = List[Dict{'A': tensor, 'B': tensor, 'block_id': int}]
        self.decomposed_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # Track block IDs for reconstruction order
        self.next_block_id = 0
    
    def add_kv(
        self,
        layer_idx: int,
        head_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """Add new key-value pair to cache.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            key: Key tensor of shape (1, head_dim) for single token
            value: Value tensor of shape (1, head_dim) for single token
        """
        if key.shape != (1, self.head_dim) or value.shape != (1, self.head_dim):
            raise ValueError(f"Expected key/value shape (1, {self.head_dim}), got {key.shape}/{value.shape}")
        
        # Add to standard cache
        self.standard_cache[layer_idx][head_idx].append((key, value))
        self.stats['total_tokens_processed'] += 1
        
        # Check if we need to compress
        if len(self.standard_cache[layer_idx][head_idx]) >= self.block_size:
            self._compress_block(layer_idx, head_idx)
    
    def _compress_block(self, layer_idx: int, head_idx: int):
        """Compress a block of tokens from standard cache to decomposed cache."""
        # Get all tokens for this layer/head
        kv_pairs = self.standard_cache[layer_idx][head_idx]
        
        if len(kv_pairs) == 0:
            return
        
        # Concatenate all keys and values
        keys = torch.cat([kv[0] for kv in kv_pairs], dim=0)  # (block_size, head_dim)
        values = torch.cat([kv[1] for kv in kv_pairs], dim=0)  # (block_size, head_dim)
        
        # Process each tensor (keys and values)
        for tensor_type, tensor in [('key', keys), ('value', values)]:
            self._compress_tensor_chunks(layer_idx, head_idx, tensor, tensor_type)
        
        # Clear standard cache for this layer/head
        self.standard_cache[layer_idx][head_idx].clear()
        
        self.stats['total_compressions'] += 1
        logger.debug(f"Compressed block for layer {layer_idx}, head {head_idx}")
    
    def _compress_tensor_chunks(
        self, 
        layer_idx: int, 
        head_idx: int, 
        tensor: torch.Tensor,
        tensor_type: str
    ):
        """Compress tensor by chunks using SVD."""
        seq_len, head_dim = tensor.shape
        
        # Split tensor into chunks along head dimension
        chunks = tensor.view(seq_len, self.n_chunks, self.chunk_size)  # (seq_len, n_chunks, chunk_size)
        
        block_id = self.next_block_id
        
        for chunk_idx in range(self.n_chunks):
            chunk = chunks[:, chunk_idx, :]  # (seq_len, chunk_size)
            rank = self.decomp_rank_map[chunk_idx]
            
            try:
                # Compress chunk
                A, B = self.compressor.compress_chunk(chunk, rank)
                
                # Store compressed chunk
                compressed_data = {
                    'A': A,
                    'B': B,
                    'block_id': block_id,
                    'tensor_type': tensor_type,
                    'original_shape': chunk.shape
                }
                
                self.decomposed_cache[layer_idx][head_idx][chunk_idx].append(compressed_data)
                
                # Optionally compute and store compression error
                if logger.isEnabledFor(logging.DEBUG):
                    error_metrics = self.compressor.compute_compression_error(chunk, A, B)
                    self.stats['compression_errors'].append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'chunk': chunk_idx,
                        'block_id': block_id,
                        'tensor_type': tensor_type,
                        **error_metrics
                    })
                
            except Exception as e:
                logger.error(f"Compression failed for layer {layer_idx}, head {head_idx}, chunk {chunk_idx}: {e}")
                raise
        
        # Increment block ID only after processing both key and value tensors
        if tensor_type == 'value':  # Assuming we process keys first, then values
            self.next_block_id += 1
    
    def get_full_kv(
        self, 
        layer_idx: int, 
        head_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full key and value tensors for a layer/head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index
            
        Returns:
            Tuple of (keys, values) tensors with full sequence
        """
        # Reconstruct from decomposed cache
        decomposed_keys, decomposed_values = self._reconstruct_decomposed_cache(layer_idx, head_idx)
        
        # Get current tokens from standard cache
        current_kv = self.standard_cache[layer_idx][head_idx]
        
        if current_kv:
            current_keys = torch.cat([kv[0] for kv in current_kv], dim=0)
            current_values = torch.cat([kv[1] for kv in current_kv], dim=0)
        else:
            current_keys = torch.empty(0, self.head_dim, device=self.device)
            current_values = torch.empty(0, self.head_dim, device=self.device)
        
        # Concatenate decomposed and current
        if decomposed_keys is not None:
            full_keys = torch.cat([decomposed_keys, current_keys], dim=0)
            full_values = torch.cat([decomposed_values, current_values], dim=0)
        else:
            full_keys = current_keys
            full_values = current_values
        
        self.stats['total_decompressions'] += 1
        
        return full_keys, full_values
    
    def _reconstruct_decomposed_cache(
        self, 
        layer_idx: int, 
        head_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Reconstruct tensors from decomposed cache."""
        decomposed = self.decomposed_cache[layer_idx][head_idx]
        
        if not decomposed or not decomposed[0]:  # No compressed data
            return None, None
        
        # Get all block IDs and sort them
        all_block_ids = set()
        for chunk_data_list in decomposed.values():
            for data in chunk_data_list:
                all_block_ids.add(data['block_id'])
        
        if not all_block_ids:
            return None, None
        
        sorted_block_ids = sorted(all_block_ids)
        
        reconstructed_keys = []
        reconstructed_values = []
        
        # Reconstruct each block in order
        for block_id in sorted_block_ids:
            key_chunks = []
            value_chunks = []
            
            # Reconstruct all chunks for this block
            for chunk_idx in range(self.n_chunks):
                chunk_data_list = decomposed[chunk_idx]
                
                # Find data for this block_id
                block_data = None
                for data in chunk_data_list:
                    if data['block_id'] == block_id:
                        block_data = data
                        break
                
                if block_data is None:
                    raise RuntimeError(f"Missing chunk {chunk_idx} for block {block_id}")
                
                # Reconstruct chunk
                reconstructed_chunk = self.compressor.decompress_chunk(
                    block_data['A'], block_data['B']
                )
                
                if block_data['tensor_type'] == 'key':
                    key_chunks.append(reconstructed_chunk)
                else:  # 'value'
                    value_chunks.append(reconstructed_chunk)
            
            # Concatenate chunks to form full tensors for this block
            if key_chunks:
                block_keys = torch.cat(key_chunks, dim=1)  # (seq_len, head_dim)
                reconstructed_keys.append(block_keys)
            
            if value_chunks:
                block_values = torch.cat(value_chunks, dim=1)  # (seq_len, head_dim)
                reconstructed_values.append(block_values)
        
        # Concatenate all blocks
        if reconstructed_keys:
            full_keys = torch.cat(reconstructed_keys, dim=0)
            full_values = torch.cat(reconstructed_values, dim=0)
            return full_keys, full_values
        
        return None, None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics."""
        standard_memory = 0
        decomposed_memory = 0
        
        # Calculate standard cache memory
        for layer_cache in self.standard_cache.values():
            for head_cache in layer_cache.values():
                for key, value in head_cache:
                    standard_memory += key.numel() + value.numel()
        
        # Calculate decomposed cache memory
        for layer_cache in self.decomposed_cache.values():
            for head_cache in layer_cache.values():
                for chunk_cache in head_cache.values():
                    for data in chunk_cache:
                        decomposed_memory += data['A'].numel() + data['B'].numel()
        
        # Calculate theoretical uncompressed memory
        total_tokens = self.stats['total_tokens_processed']
        theoretical_memory = total_tokens * self.head_dim * self.n_layers * self.n_heads * 2  # *2 for key+value
        
        compression_ratio = decomposed_memory / max(1, theoretical_memory - standard_memory * 4)  # Rough estimate
        
        return {
            'standard_cache_elements': standard_memory,
            'decomposed_cache_elements': decomposed_memory,
            'total_elements': standard_memory + decomposed_memory,
            'theoretical_uncompressed_elements': theoretical_memory,
            'compression_ratio': compression_ratio,
            'total_tokens_processed': total_tokens,
            'compression_count': self.stats['total_compressions'],
            'decompression_count': self.stats['total_decompressions']
        }
    
    def clear_cache(self):
        """Clear all cache data."""
        self.standard_cache.clear()
        self.decomposed_cache.clear()
        self.next_block_id = 0
        self.stats = {
            'total_tokens_processed': 0,
            'total_compressions': 0,
            'total_decompressions': 0,
            'compression_errors': []
        }
