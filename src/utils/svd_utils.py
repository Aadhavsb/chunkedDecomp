"""SVD utilities for chunked decomposition."""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SVDCompressor:
    """Handles SVD-based compression and decompression for tensor chunks."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize SVD compressor.
        
        Args:
            device: Device to perform computations on. If None, uses current device.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compress_chunk(
        self, 
        tensor: torch.Tensor, 
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a tensor chunk using SVD.
        
        Args:
            tensor: Input tensor of shape (sequence_length, chunk_size)
            rank: Target rank for compression
            
        Returns:
            Tuple of (A, B) matrices where A @ B.T approximates the original tensor
            A: shape (sequence_length, rank)
            B: shape (chunk_size, rank)
        """
        if tensor.numel() == 0:
            raise ValueError("Cannot compress empty tensor")
            
        if rank <= 0:
            raise ValueError(f"Rank must be positive, got {rank}")
            
        if rank > min(tensor.shape):
            logger.warning(f"Rank {rank} exceeds tensor dimensions {tensor.shape}, using max possible rank")
            rank = min(tensor.shape)
        
        # Perform SVD
        try:
            U, S, V = torch.svd(tensor)
        except RuntimeError as e:
            logger.error(f"SVD failed for tensor shape {tensor.shape}: {e}")
            raise
        
        # Truncate to target rank
        U_truncated = U[:, :rank]  # (sequence_length, rank)
        S_truncated = S[:rank]     # (rank,)
        V_truncated = V[:, :rank]  # (chunk_size, rank)
        
        # Create compressed matrices
        A = U_truncated * S_truncated.unsqueeze(0)  # (sequence_length, rank)
        B = V_truncated                             # (chunk_size, rank)
        
        return A, B
    
    def decompress_chunk(
        self, 
        A: torch.Tensor, 
        B: torch.Tensor
    ) -> torch.Tensor:
        """Decompress a tensor chunk from SVD matrices.
        
        Args:
            A: Left matrix of shape (sequence_length, rank)
            B: Right matrix of shape (chunk_size, rank)
            
        Returns:
            Reconstructed tensor of shape (sequence_length, chunk_size)
        """
        if A.shape[1] != B.shape[1]:
            raise ValueError(f"Rank mismatch: A has rank {A.shape[1]}, B has rank {B.shape[1]}")
        
        return A @ B.T
    
    def compute_compression_error(
        self, 
        original: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor
    ) -> Dict[str, float]:
        """Compute various error metrics for compression quality.
        
        Args:
            original: Original tensor
            A, B: Compressed matrices
            
        Returns:
            Dictionary with error metrics
        """
        reconstructed = self.decompress_chunk(A, B)
        
        # Compute various error metrics
        diff = original - reconstructed
        
        frobenius_error = torch.norm(diff, 'fro').item()
        relative_error = frobenius_error / torch.norm(original, 'fro').item()
        max_error = torch.max(torch.abs(diff)).item()
        
        return {
            'frobenius_error': frobenius_error,
            'relative_error': relative_error,
            'max_error': max_error,
            'mse': torch.mean(diff ** 2).item()
        }
    
    def optimal_rank_analysis(
        self, 
        tensor: torch.Tensor, 
        target_compression_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """Analyze optimal rank for given compression ratio.
        
        Args:
            tensor: Input tensor to analyze
            target_compression_ratio: Desired compression ratio (0 < ratio < 1)
            
        Returns:
            Analysis results including recommended rank and error estimates
        """
        seq_len, chunk_size = tensor.shape
        original_elements = seq_len * chunk_size
        
        # Calculate rank that achieves target compression ratio
        # Compressed size = seq_len * rank + chunk_size * rank
        # Compression ratio = compressed_size / original_size
        target_compressed_size = int(original_elements * target_compression_ratio)
        optimal_rank = target_compressed_size // (seq_len + chunk_size)
        optimal_rank = max(1, min(optimal_rank, min(seq_len, chunk_size)))
        
        # Perform SVD to get singular values for analysis
        try:
            _, S, _ = torch.svd(tensor)
        except RuntimeError:
            return {
                'recommended_rank': optimal_rank,
                'error': 'SVD failed',
                'singular_values': None
            }
        
        # Calculate error for the recommended rank
        if optimal_rank < len(S):
            truncated_error = torch.sum(S[optimal_rank:] ** 2).item()
            total_energy = torch.sum(S ** 2).item()
            relative_truncation_error = truncated_error / total_energy if total_energy > 0 else 0
        else:
            relative_truncation_error = 0
        
        return {
            'recommended_rank': optimal_rank,
            'actual_compression_ratio': (seq_len * optimal_rank + chunk_size * optimal_rank) / original_elements,
            'relative_truncation_error': relative_truncation_error,
            'singular_values': S.cpu().numpy(),
            'rank_energy_ratio': torch.sum(S[:optimal_rank] ** 2).item() / torch.sum(S ** 2).item()
        }


def batch_compress_chunks(
    chunks: List[torch.Tensor], 
    ranks: List[int], 
    compressor: Optional[SVDCompressor] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Compress multiple chunks with different ranks.
    
    Args:
        chunks: List of tensor chunks to compress
        ranks: List of target ranks for each chunk
        compressor: SVD compressor instance (creates new if None)
        
    Returns:
        List of (A, B) matrix pairs
    """
    if len(chunks) != len(ranks):
        raise ValueError(f"Number of chunks ({len(chunks)}) must match number of ranks ({len(ranks)})")
    
    if compressor is None:
        compressor = SVDCompressor()
    
    compressed = []
    for chunk, rank in zip(chunks, ranks):
        A, B = compressor.compress_chunk(chunk, rank)
        compressed.append((A, B))
    
    return compressed


def create_adaptive_rank_map(
    n_chunks: int,
    chunk_size: int,
    strategy: str = "uniform",
    base_compression_ratio: float = 0.5,
    **kwargs
) -> Dict[int, int]:
    """Create rank map for adaptive compression.
    
    Args:
        n_chunks: Number of chunks
        chunk_size: Size of each chunk
        strategy: Compression strategy ('uniform', 'progressive', 'adaptive')
        base_compression_ratio: Base compression ratio
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        Dictionary mapping chunk index to rank
    """
    if strategy == "uniform":
        base_rank = max(1, int(chunk_size * base_compression_ratio))
        return {i: base_rank for i in range(n_chunks)}
    
    elif strategy == "progressive":
        start_ratio = kwargs.get('start_ratio', 0.75)
        end_ratio = kwargs.get('end_ratio', 0.25)
        
        rank_map = {}
        for i in range(n_chunks):
            ratio = start_ratio - (start_ratio - end_ratio) * (i / max(1, n_chunks - 1))
            rank = max(1, int(chunk_size * ratio))
            rank_map[i] = rank
        return rank_map
    
    elif strategy == "adaptive":
        # This would be updated based on actual importance analysis
        # For now, use progressive as fallback
        return create_adaptive_rank_map(n_chunks, chunk_size, "progressive", **kwargs)
    
    else:
        raise ValueError(f"Unknown compression strategy: {strategy}")
