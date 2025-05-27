#!/usr/bin/env python3
"""
Unit tests for SVD compression utilities

Tests cover:
- SVD compression and decompression
- Chunked decomposition
- Adaptive rank selection
- Error analysis
- Memory efficiency
- Different matrix configurations
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.svd_utils import SVDCompressor


class TestSVDCompressor(unittest.TestCase):
    """Test cases for SVDCompressor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compressor = SVDCompressor(device=self.device)
        
        # Create test matrices of different sizes
        self.small_matrix = torch.randn(32, 64, device=self.device)
        self.medium_matrix = torch.randn(128, 256, device=self.device)
        self.large_matrix = torch.randn(512, 1024, device=self.device)
        self.square_matrix = torch.randn(256, 256, device=self.device)
        
        # Non-square matrices
        self.tall_matrix = torch.randn(1024, 128, device=self.device)
        self.wide_matrix = torch.randn(128, 1024, device=self.device)
    
    def test_basic_compression_decompression(self):
        """Test basic SVD compression and decompression."""
        compression_ratio = 0.5
        
        # Compress the matrix
        compressed_data = self.compressor.compress_matrix(
            self.medium_matrix, 
            compression_ratio=compression_ratio
        )
        
        # Check compressed data structure
        self.assertIn('U', compressed_data)
        self.assertIn('S', compressed_data)
        self.assertIn('V', compressed_data)
        self.assertIn('rank', compressed_data)
        self.assertIn('original_shape', compressed_data)
        self.assertIn('compression_ratio', compressed_data)
        
        # Decompress the matrix
        decompressed_matrix = self.compressor.decompress_matrix(compressed_data)
        
        # Check shape preservation
        self.assertEqual(decompressed_matrix.shape, self.medium_matrix.shape)
        
        # Check approximation quality
        reconstruction_error = torch.norm(self.medium_matrix - decompressed_matrix)
        original_norm = torch.norm(self.medium_matrix)
        relative_error = reconstruction_error / original_norm
        
        # Error should be reasonable for 50% compression
        self.assertLess(relative_error, 0.5)
    
    def test_different_compression_ratios(self):
        """Test compression with different ratios."""
        ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for ratio in ratios:
            with self.subTest(ratio=ratio):
                compressed_data = self.compressor.compress_matrix(
                    self.medium_matrix,
                    compression_ratio=ratio
                )
                
                # Check rank is approximately correct
                expected_rank = int(min(self.medium_matrix.shape) * ratio)
                actual_rank = compressed_data['rank']
                
                # Allow some tolerance
                self.assertLessEqual(abs(actual_rank - expected_rank), 5)
                
                # Check compression ratio
                self.assertAlmostEqual(
                    compressed_data['compression_ratio'],
                    ratio,
                    delta=0.1
                )
    
    def test_chunked_compression(self):
        """Test chunked compression for large matrices."""
        chunk_size = 64
        compression_ratio = 0.5
        
        # Compress with chunking
        compressed_data = self.compressor.compress_matrix(
            self.large_matrix,
            compression_ratio=compression_ratio,
            chunk_size=chunk_size
        )
        
        # Should contain chunk information
        self.assertIn('is_chunked', compressed_data)
        self.assertIn('chunks', compressed_data)
        self.assertTrue(compressed_data['is_chunked'])
        
        # Decompress
        decompressed_matrix = self.compressor.decompress_matrix(compressed_data)
        
        # Check shape preservation
        self.assertEqual(decompressed_matrix.shape, self.large_matrix.shape)
        
        # Check approximation quality
        reconstruction_error = torch.norm(self.large_matrix - decompressed_matrix)
        original_norm = torch.norm(self.large_matrix)
        relative_error = reconstruction_error / original_norm
        
        self.assertLess(relative_error, 0.7)  # Chunked compression may be less accurate
    
    def test_adaptive_rank_selection(self):
        """Test adaptive rank selection based on error threshold."""
        error_threshold = 0.1
        
        compressed_data = self.compressor.compress_matrix(
            self.medium_matrix,
            error_threshold=error_threshold,
            adaptive_rank=True
        )
        
        # Check adaptive rank was used
        self.assertIn('adaptive_rank_used', compressed_data)
        self.assertTrue(compressed_data['adaptive_rank_used'])
        
        # Decompress and check error
        decompressed_matrix = self.compressor.decompress_matrix(compressed_data)
        
        reconstruction_error = torch.norm(self.medium_matrix - decompressed_matrix)
        original_norm = torch.norm(self.medium_matrix)
        relative_error = reconstruction_error / original_norm
        
        # Error should be below threshold (with some tolerance)
        self.assertLess(relative_error, error_threshold * 1.5)
    
    def test_non_square_matrices(self):
        """Test compression of non-square matrices."""
        matrices = [
            ('tall', self.tall_matrix),
            ('wide', self.wide_matrix)
        ]
        
        for name, matrix in matrices:
            with self.subTest(matrix_type=name):
                compressed_data = self.compressor.compress_matrix(
                    matrix,
                    compression_ratio=0.5
                )
                
                decompressed_matrix = self.compressor.decompress_matrix(compressed_data)
                
                # Check shape preservation
                self.assertEqual(decompressed_matrix.shape, matrix.shape)
                
                # Check reconstruction quality
                reconstruction_error = torch.norm(matrix - decompressed_matrix)
                original_norm = torch.norm(matrix)
                relative_error = reconstruction_error / original_norm
                
                self.assertLess(relative_error, 0.6)
    
    def test_compression_statistics(self):
        """Test compression statistics collection."""
        compressed_data = self.compressor.compress_matrix(
            self.medium_matrix,
            compression_ratio=0.5,
            collect_stats=True
        )
        
        # Check statistics are present
        self.assertIn('compression_stats', compressed_data)
        stats = compressed_data['compression_stats']
        
        required_stats = [
            'original_size',
            'compressed_size',
            'memory_reduction_mb',
            'compression_time_ms',
            'singular_values_kept',
            'singular_values_discarded'
        ]
        
        for stat in required_stats:
            self.assertIn(stat, stats)
        
        # Check values are reasonable
        self.assertGreater(stats['original_size'], stats['compressed_size'])
        self.assertGreater(stats['memory_reduction_mb'], 0)
        self.assertGreater(stats['compression_time_ms'], 0)
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        compressed_data = self.compressor.compress_matrix(
            self.medium_matrix,
            compression_ratio=0.5
        )
        
        error_analysis = self.compressor.analyze_compression_error(
            self.medium_matrix,
            compressed_data
        )
        
        # Check error metrics
        required_metrics = [
            'frobenius_error',
            'relative_error',
            'spectral_error',
            'max_absolute_error',
            'mean_absolute_error'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, error_analysis)
            self.assertGreaterEqual(error_analysis[metric], 0)
        
        # Check relative error is reasonable
        self.assertLess(error_analysis['relative_error'], 1.0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of compression."""
        # Test with different matrix sizes
        matrices = [
            self.small_matrix,
            self.medium_matrix,
            self.large_matrix
        ]
        
        for matrix in matrices:
            with self.subTest(shape=matrix.shape):
                # Calculate original memory usage
                original_memory = matrix.numel() * matrix.element_size()
                
                # Compress matrix
                compressed_data = self.compressor.compress_matrix(
                    matrix,
                    compression_ratio=0.5,
                    collect_stats=True
                )
                
                # Calculate compressed memory usage
                stats = compressed_data['compression_stats']
                compressed_memory = stats['compressed_size']
                
                # Check memory reduction
                memory_reduction = (original_memory - compressed_memory) / original_memory
                self.assertGreater(memory_reduction, 0.2)  # At least 20% reduction
    
    def test_batch_compression(self):
        """Test batch compression of multiple matrices."""
        matrices = [
            torch.randn(64, 128, device=self.device),
            torch.randn(128, 64, device=self.device),
            torch.randn(96, 96, device=self.device)
        ]
        
        # Compress batch
        compressed_batch = self.compressor.compress_batch(
            matrices,
            compression_ratio=0.5
        )
        
        # Check batch structure
        self.assertIsInstance(compressed_batch, list)
        self.assertEqual(len(compressed_batch), len(matrices))
        
        # Decompress batch
        decompressed_batch = self.compressor.decompress_batch(compressed_batch)
        
        # Check shapes and quality
        for i, (original, decompressed) in enumerate(zip(matrices, decompressed_batch)):
            with self.subTest(matrix_index=i):
                self.assertEqual(original.shape, decompressed.shape)
                
                reconstruction_error = torch.norm(original - decompressed)
                original_norm = torch.norm(original)
                relative_error = reconstruction_error / original_norm
                
                self.assertLess(relative_error, 0.6)
    
    def test_rank_mapping(self):
        """Test adaptive rank mapping functionality."""
        # Test with different error thresholds
        thresholds = [0.05, 0.1, 0.2, 0.3]
        
        for threshold in thresholds:
            with self.subTest(threshold=threshold):
                optimal_rank = self.compressor.find_optimal_rank(
                    self.medium_matrix,
                    error_threshold=threshold
                )
                
                # Check rank is reasonable
                min_dim = min(self.medium_matrix.shape)
                self.assertGreater(optimal_rank, 0)
                self.assertLessEqual(optimal_rank, min_dim)
                
                # Test compression with this rank
                compressed_data = self.compressor.compress_matrix(
                    self.medium_matrix,
                    rank=optimal_rank
                )
                
                # Check error is below threshold
                error_analysis = self.compressor.analyze_compression_error(
                    self.medium_matrix,
                    compressed_data
                )
                
                # Allow some tolerance
                self.assertLess(error_analysis['relative_error'], threshold * 1.5)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test invalid compression ratio
        with self.assertRaises(ValueError):
            self.compressor.compress_matrix(
                self.medium_matrix,
                compression_ratio=1.5  # Invalid ratio > 1
            )
        
        with self.assertRaises(ValueError):
            self.compressor.compress_matrix(
                self.medium_matrix,
                compression_ratio=-0.1  # Invalid negative ratio
            )
        
        # Test invalid rank
        with self.assertRaises(ValueError):
            self.compressor.compress_matrix(
                self.medium_matrix,
                rank=1000  # Rank larger than matrix dimensions
            )
        
        # Test empty matrix
        empty_matrix = torch.empty(0, 0, device=self.device)
        with self.assertRaises(ValueError):
            self.compressor.compress_matrix(empty_matrix)
    
    def test_compression_consistency(self):
        """Test compression consistency across multiple runs."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Create deterministic matrix
        test_matrix = torch.randn(100, 200, device=self.device)
        
        # Compress multiple times
        compressed_results = []
        for _ in range(3):
            torch.manual_seed(42)  # Reset seed
            compressed_data = self.compressor.compress_matrix(
                test_matrix,
                compression_ratio=0.5
            )
            compressed_results.append(compressed_data)
        
        # Check consistency
        for i in range(1, len(compressed_results)):
            # Ranks should be the same
            self.assertEqual(
                compressed_results[0]['rank'],
                compressed_results[i]['rank']
            )
            
            # Compression ratios should be the same
            self.assertAlmostEqual(
                compressed_results[0]['compression_ratio'],
                compressed_results[i]['compression_ratio'],
                delta=0.01
            )
    
    def test_special_matrices(self):
        """Test compression of special matrices (low rank, identity, etc.)."""
        # Low rank matrix
        U = torch.randn(100, 5, device=self.device)
        V = torch.randn(5, 200, device=self.device)
        low_rank_matrix = U @ V
        
        compressed_data = self.compressor.compress_matrix(
            low_rank_matrix,
            compression_ratio=0.3
        )
        
        decompressed_matrix = self.compressor.decompress_matrix(compressed_data)
        
        # Should have very good reconstruction for low-rank matrix
        reconstruction_error = torch.norm(low_rank_matrix - decompressed_matrix)
        original_norm = torch.norm(low_rank_matrix)
        relative_error = reconstruction_error / original_norm
        
        self.assertLess(relative_error, 0.1)  # Very low error expected
        
        # Identity matrix
        identity_matrix = torch.eye(100, device=self.device)
        
        compressed_data = self.compressor.compress_matrix(
            identity_matrix,
            compression_ratio=0.5
        )
        
        decompressed_matrix = self.compressor.decompress_matrix(compressed_data)
        
        # Should reconstruct identity well
        reconstruction_error = torch.norm(identity_matrix - decompressed_matrix)
        self.assertLess(reconstruction_error, 0.5)


class TestCompressionIntegration(unittest.TestCase):
    """Integration tests for compression with neural network layers."""
    
    def setUp(self):
        """Set up test fixtures with neural network layers."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compressor = SVDCompressor(device=self.device)
        
        # Create test layers
        self.linear_layer = nn.Linear(256, 128)
        self.large_linear = nn.Linear(1024, 512)
        self.conv_layer = nn.Conv2d(64, 128, kernel_size=3)
        
        # Move to device
        self.linear_layer.to(self.device)
        self.large_linear.to(self.device)
        self.conv_layer.to(self.device)
    
    def test_linear_layer_compression(self):
        """Test compression of linear layer weights."""
        # Get original weight
        original_weight = self.linear_layer.weight.data.clone()
        original_output = None
        
        # Test input
        test_input = torch.randn(4, 256, device=self.device)
        
        with torch.no_grad():
            original_output = self.linear_layer(test_input)
        
        # Compress weight matrix
        compressed_data = self.compressor.compress_matrix(
            original_weight,
            compression_ratio=0.5
        )
        
        # Decompress and replace weight
        compressed_weight = self.compressor.decompress_matrix(compressed_data)
        self.linear_layer.weight.data = compressed_weight
        
        # Test compressed layer
        with torch.no_grad():
            compressed_output = self.linear_layer(test_input)
        
        # Check output shapes match
        self.assertEqual(original_output.shape, compressed_output.shape)
        
        # Check outputs are reasonably similar
        output_diff = torch.norm(original_output - compressed_output)
        output_norm = torch.norm(original_output)
        relative_diff = output_diff / output_norm
        
        self.assertLess(relative_diff, 0.7)  # Allow reasonable difference
    
    def test_layer_wise_compression(self):
        """Test compression of entire model layer by layer."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(self.device)
        
        # Test input
        test_input = torch.randn(4, 128, device=self.device)
        
        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(test_input)
        
        # Compress linear layers
        compression_stats = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                original_weight = module.weight.data.clone()
                
                compressed_data = self.compressor.compress_matrix(
                    original_weight,
                    compression_ratio=0.5,
                    collect_stats=True
                )
                
                compressed_weight = self.compressor.decompress_matrix(compressed_data)
                module.weight.data = compressed_weight
                
                compression_stats.append(compressed_data['compression_stats'])
        
        # Test compressed model
        with torch.no_grad():
            compressed_output = model(test_input)
        
        # Check functionality is preserved
        self.assertEqual(original_output.shape, compressed_output.shape)
        
        # Check compression actually happened
        self.assertGreater(len(compression_stats), 0)
        
        # Check memory reduction
        total_memory_reduction = sum(
            stats['memory_reduction_mb'] for stats in compression_stats
        )
        self.assertGreater(total_memory_reduction, 0)
    
    def test_conv_layer_compression(self):
        """Test compression of convolutional layer weights."""
        # Get original weight (reshape for SVD)
        original_weight = self.conv_layer.weight.data.clone()
        out_channels, in_channels, kh, kw = original_weight.shape
        
        # Reshape to 2D matrix
        weight_2d = original_weight.view(out_channels, -1)
        
        # Test input
        test_input = torch.randn(2, 64, 32, 32, device=self.device)
        
        with torch.no_grad():
            original_output = self.conv_layer(test_input)
        
        # Compress reshaped weight
        compressed_data = self.compressor.compress_matrix(
            weight_2d,
            compression_ratio=0.6
        )
        
        # Decompress and reshape back
        compressed_weight_2d = self.compressor.decompress_matrix(compressed_data)
        compressed_weight = compressed_weight_2d.view(out_channels, in_channels, kh, kw)
        
        # Replace weight
        self.conv_layer.weight.data = compressed_weight
        
        # Test compressed layer
        with torch.no_grad():
            compressed_output = self.conv_layer(test_input)
        
        # Check output shapes match
        self.assertEqual(original_output.shape, compressed_output.shape)
        
        # Check outputs are reasonably similar
        output_diff = torch.norm(original_output - compressed_output)
        output_norm = torch.norm(original_output)
        relative_diff = output_diff / output_norm
        
        self.assertLess(relative_diff, 0.8)  # Conv layers might have more variation


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
