#!/usr/bin/env python3
"""
Unit tests for ChunkedDecomp main class

Tests cover:
- Model initialization and configuration
- Compression application and reversal
- KV cache integration
- Memory management
- Error handling
- Configuration validation
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil
import yaml
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.chunked_decomp import ChunkedDecomp
from models.kv_cache import ChunkedKVCache
from utils.svd_utils import SVDCompressor


class TestChunkedDecomp(unittest.TestCase):
    """Test cases for ChunkedDecomp class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create a simple test model
        self.test_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        )
        
        # Basic configuration
        self.basic_config = {
            'compression': {
                'compression_ratio': 0.5,
                'chunk_size': 32,
                'adaptive_rank': True,
                'min_rank': 4,
                'max_rank': 64
            },
            'kv_cache': {
                'max_cache_size': 1000000,
                'compression_threshold': 0.8,
                'enable_chunked_cache': True
            },
            'optimization': {
                'memory_efficient': True,
                'batch_processing': True
            }
        }
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ChunkedDecomp initialization."""
        # Test basic initialization
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        self.assertIsNotNone(chunked_decomp.model)
        self.assertIsNotNone(chunked_decomp.config)
        self.assertIsNotNone(chunked_decomp.compressor)
        self.assertIsInstance(chunked_decomp.compressor, SVDCompressor)
        
        # Test device assignment
        self.assertEqual(chunked_decomp.device, self.device)
        
        # Test configuration loading
        self.assertEqual(
            chunked_decomp.config['compression']['compression_ratio'], 
            0.5
        )
    
    def test_initialization_with_invalid_config(self):
        """Test initialization with invalid configuration."""
        invalid_config = {
            'compression': {
                'compression_ratio': 1.5,  # Invalid ratio > 1
                'chunk_size': -10  # Invalid negative chunk size
            }
        }
        
        with self.assertRaises(ValueError):
            ChunkedDecomp(
                model=self.test_model,
                config=invalid_config
            )
    
    def test_compression_application(self):
        """Test compression application to model."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        # Get original parameter count
        original_params = sum(p.numel() for p in self.test_model.parameters())
        
        # Apply compression
        compression_stats = chunked_decomp.apply_compression()
        
        # Check compression was applied
        self.assertIsNotNone(compression_stats)
        self.assertIn('layers_compressed', compression_stats)
        self.assertIn('compression_ratio', compression_stats)
        self.assertIn('memory_reduction_mb', compression_stats)
        
        # Check model still works
        test_input = torch.randn(4, 64)
        output = chunked_decomp.model(test_input)
        self.assertEqual(output.shape, (4, 32))
    
    def test_compression_with_different_ratios(self):
        """Test compression with different ratios."""
        ratios = [0.3, 0.5, 0.7]
        
        for ratio in ratios:
            with self.subTest(ratio=ratio):
                config = self.basic_config.copy()
                config['compression']['compression_ratio'] = ratio
                
                chunked_decomp = ChunkedDecomp(
                    model=self.test_model,
                    config=config
                )
                
                stats = chunked_decomp.apply_compression()
                
                # Check compression ratio is approximately correct
                self.assertAlmostEqual(
                    stats['compression_ratio'], 
                    ratio, 
                    delta=0.1
                )
    
    def test_kv_cache_integration(self):
        """Test KV cache integration."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        # Enable KV cache
        chunked_decomp.enable_kv_cache()
        
        self.assertIsNotNone(chunked_decomp.kv_cache)
        self.assertIsInstance(chunked_decomp.kv_cache, ChunkedKVCache)
        
        # Test cache operations
        test_key = "test_layer"
        test_tensor = torch.randn(4, 64, 128)
        
        # Store in cache
        chunked_decomp.kv_cache.store(test_key, test_tensor, test_tensor)
        
        # Retrieve from cache
        cached_k, cached_v = chunked_decomp.kv_cache.get(test_key)
        
        self.assertIsNotNone(cached_k)
        self.assertIsNotNone(cached_v)
        
        # Disable cache
        chunked_decomp.disable_kv_cache()
        self.assertIsNone(chunked_decomp.kv_cache)
    
    def test_memory_management(self):
        """Test memory management features."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        # Test memory statistics
        memory_stats = chunked_decomp.get_memory_stats()
        
        self.assertIn('model_memory_mb', memory_stats)
        self.assertIn('cache_memory_mb', memory_stats)
        self.assertIn('total_memory_mb', memory_stats)
        self.assertIn('device', memory_stats)
        
        # Test memory cleanup
        chunked_decomp.clear_cache()
        
        # Memory stats should show reduced cache usage
        new_stats = chunked_decomp.get_memory_stats()
        self.assertLessEqual(
            new_stats['cache_memory_mb'], 
            memory_stats['cache_memory_mb']
        )
    
    def test_model_serialization(self):
        """Test model saving and loading."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        # Apply compression
        chunked_decomp.apply_compression()
        
        # Save model
        save_path = os.path.join(self.temp_dir, 'test_model.pt')
        save_stats = chunked_decomp.save_model(save_path)
        
        self.assertTrue(os.path.exists(save_path))
        self.assertIn('file_size_mb', save_stats)
        self.assertIn('compression_stats', save_stats)
        
        # Test model can be loaded back
        self.assertTrue(os.path.getsize(save_path) > 0)
    
    def test_compression_reversal(self):
        """Test compression reversal functionality."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        # Get original output
        test_input = torch.randn(2, 64)
        original_output = self.test_model(test_input)
        
        # Apply compression
        chunked_decomp.apply_compression()
        compressed_output = chunked_decomp.model(test_input)
        
        # Reverse compression (if implemented)
        try:
            chunked_decomp.reverse_compression()
            reversed_output = chunked_decomp.model(test_input)
            
            # Output should be closer to original
            original_diff = torch.norm(original_output - compressed_output)
            reversed_diff = torch.norm(original_output - reversed_output)
            
            self.assertLess(reversed_diff, original_diff)
            
        except NotImplementedError:
            # Reversal not implemented - that's OK
            pass
    
    def test_adaptive_compression(self):
        """Test adaptive compression features."""
        # Enable adaptive rank
        config = self.basic_config.copy()
        config['compression']['adaptive_rank'] = True
        config['compression']['error_threshold'] = 0.1
        
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=config
        )
        
        stats = chunked_decomp.apply_compression()
        
        # Check adaptive features were used
        self.assertIn('adaptive_stats', stats)
        adaptive_stats = stats['adaptive_stats']
        
        self.assertIn('layers_with_adaptive_rank', adaptive_stats)
        self.assertIn('rank_adjustments', adaptive_stats)
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        chunked_decomp.apply_compression()
        
        # Test with different batch sizes
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                test_input = torch.randn(batch_size, 64)
                output = chunked_decomp.model(test_input)
                
                self.assertEqual(output.shape, (batch_size, 32))
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with None model
        with self.assertRaises(ValueError):
            ChunkedDecomp(model=None, config=self.basic_config)
        
        # Test with invalid compression ratio
        invalid_config = self.basic_config.copy()
        invalid_config['compression']['compression_ratio'] = 2.0
        
        with self.assertRaises(ValueError):
            ChunkedDecomp(model=self.test_model, config=invalid_config)
        
        # Test compression on already compressed model
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        chunked_decomp.apply_compression()
        
        # Second compression should handle gracefully
        stats2 = chunked_decomp.apply_compression()
        self.assertIsNotNone(stats2)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        # Test valid configuration
        is_valid = chunked_decomp.validate_config()
        self.assertTrue(is_valid)
        
        # Test configuration updates
        new_config = {
            'compression': {
                'compression_ratio': 0.3,
                'chunk_size': 64
            }
        }
        
        chunked_decomp.update_config(new_config)
        
        self.assertEqual(
            chunked_decomp.config['compression']['compression_ratio'], 
            0.3
        )
        self.assertEqual(
            chunked_decomp.config['compression']['chunk_size'], 
            64
        )
    
    def test_compression_statistics(self):
        """Test compression statistics collection."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        stats = chunked_decomp.apply_compression()
        
        # Check required statistics
        required_stats = [
            'layers_compressed',
            'total_layers',
            'compression_ratio',
            'memory_reduction_mb',
            'parameter_reduction_count',
            'compression_time_seconds'
        ]
        
        for stat in required_stats:
            self.assertIn(stat, stats)
            self.assertIsNotNone(stats[stat])
        
        # Check statistics values are reasonable
        self.assertGreater(stats['layers_compressed'], 0)
        self.assertGreaterEqual(stats['total_layers'], stats['layers_compressed'])
        self.assertGreater(stats['compression_ratio'], 0)
        self.assertLess(stats['compression_ratio'], 1)
        self.assertGreaterEqual(stats['memory_reduction_mb'], 0)
        self.assertGreater(stats['compression_time_seconds'], 0)
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        # Enable performance monitoring
        chunked_decomp.enable_performance_monitoring()
        
        # Apply compression with monitoring
        stats = chunked_decomp.apply_compression()
        
        # Check monitoring data
        self.assertIn('performance_metrics', stats)
        perf_metrics = stats['performance_metrics']
        
        self.assertIn('cpu_usage_percent', perf_metrics)
        self.assertIn('memory_usage_mb', perf_metrics)
        self.assertIn('gpu_utilization_percent', perf_metrics)
        
        # Run inference with monitoring
        test_input = torch.randn(4, 64)
        output = chunked_decomp.model(test_input)
        
        perf_data = chunked_decomp.get_performance_data()
        self.assertIsNotNone(perf_data)
    
    @patch('torch.save')
    def test_save_with_compression_metadata(self, mock_save):
        """Test saving model with compression metadata."""
        chunked_decomp = ChunkedDecomp(
            model=self.test_model,
            config=self.basic_config
        )
        
        chunked_decomp.apply_compression()
        
        save_path = os.path.join(self.temp_dir, 'test_model.pt')
        chunked_decomp.save_model(save_path, include_metadata=True)
        
        # Check that save was called
        mock_save.assert_called_once()
        
        # Check the saved data structure
        call_args = mock_save.call_args
        saved_data = call_args[0][0]  # First argument to torch.save
        
        self.assertIn('model_state_dict', saved_data)
        self.assertIn('compression_config', saved_data)
        self.assertIn('compression_stats', saved_data)
        self.assertIn('metadata', saved_data)


class TestChunkedDecompIntegration(unittest.TestCase):
    """Integration tests for ChunkedDecomp with real transformer components."""
    
    def setUp(self):
        """Set up test fixtures with transformer-like components."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create a transformer-like model
        class TransformerBlock(nn.Module):
            def __init__(self, hidden_size=128, num_heads=8):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.mlp = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
            
            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                
                # MLP
                mlp_out = self.mlp(x)
                x = self.norm2(x + mlp_out)
                
                return x
        
        self.transformer_model = nn.Sequential(
            nn.Embedding(1000, 128),
            TransformerBlock(128, 8),
            TransformerBlock(128, 8),
            nn.Linear(128, 1000)
        )
        
        self.config = {
            'compression': {
                'compression_ratio': 0.5,
                'chunk_size': 64,
                'adaptive_rank': True,
                'layers_to_compress': ['Linear']  # Only compress linear layers
            },
            'kv_cache': {
                'max_cache_size': 2000000,
                'compression_threshold': 0.7
            }
        }
    
    def test_transformer_compression(self):
        """Test compression on transformer-like model."""
        chunked_decomp = ChunkedDecomp(
            model=self.transformer_model,
            config=self.config
        )
        
        # Test input (token IDs)
        test_input = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
        
        # Get original output
        self.transformer_model.eval()
        with torch.no_grad():
            original_output = self.transformer_model(test_input)
        
        # Apply compression
        stats = chunked_decomp.apply_compression()
        
        # Check compression was applied
        self.assertGreater(stats['layers_compressed'], 0)
        
        # Test compressed model still works
        chunked_decomp.model.eval()
        with torch.no_grad():
            compressed_output = chunked_decomp.model(test_input)
        
        # Outputs should have same shape
        self.assertEqual(original_output.shape, compressed_output.shape)
        
        # Outputs should be reasonably similar
        output_diff = torch.norm(original_output - compressed_output)
        self.assertLess(output_diff, 100.0)  # Reasonable threshold
    
    def test_kv_cache_with_transformer(self):
        """Test KV cache integration with transformer model."""
        chunked_decomp = ChunkedDecomp(
            model=self.transformer_model,
            config=self.config
        )
        
        # Enable KV cache
        chunked_decomp.enable_kv_cache()
        
        # Apply compression
        chunked_decomp.apply_compression()
        
        # Test multiple forward passes (simulating generation)
        test_inputs = [
            torch.randint(0, 1000, (1, 5)),
            torch.randint(0, 1000, (1, 7)),
            torch.randint(0, 1000, (1, 3))
        ]
        
        for i, test_input in enumerate(test_inputs):
            with torch.no_grad():
                output = chunked_decomp.model(test_input)
                self.assertEqual(output.shape[-1], 1000)  # Vocab size
        
        # Check cache statistics
        cache_stats = chunked_decomp.kv_cache.get_stats()
        self.assertGreater(cache_stats['total_stores'], 0)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
