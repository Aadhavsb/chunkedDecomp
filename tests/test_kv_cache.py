#!/usr/bin/env python3
"""
Unit tests for ChunkedKVCache

Tests cover:
- Cache storage and retrieval
- Automatic compression triggers
- Memory management
- Cache statistics
- Multi-layer caching
- Cache persistence
"""

import unittest
import torch
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.kv_cache import ChunkedKVCache
from utils.svd_utils import SVDCompressor


class TestChunkedKVCache(unittest.TestCase):
    """Test cases for ChunkedKVCache class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Basic cache configuration
        self.cache_config = {
            'max_cache_size': 1000000,  # 1MB
            'compression_threshold': 0.8,
            'enable_chunked_cache': True,
            'compression_ratio': 0.5,
            'chunk_size': 64
        }
        
        self.cache = ChunkedKVCache(
            config=self.cache_config,
            device=self.device
        )
        
        # Create test tensors
        self.test_key_tensor = torch.randn(4, 8, 64, device=self.device)  # [batch, heads, seq, dim]
        self.test_value_tensor = torch.randn(4, 8, 64, device=self.device)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        self.assertIsNotNone(self.cache.compressor)
        self.assertIsInstance(self.cache.compressor, SVDCompressor)
        self.assertEqual(self.cache.device, self.device)
        self.assertEqual(self.cache.config['max_cache_size'], 1000000)
    
    def test_basic_store_and_get(self):
        """Test basic store and retrieve operations."""
        layer_name = "layer_0"
        
        # Store KV pair
        self.cache.store(layer_name, self.test_key_tensor, self.test_value_tensor)
        
        # Retrieve KV pair
        cached_k, cached_v = self.cache.get(layer_name)
        
        # Check retrieval was successful
        self.assertIsNotNone(cached_k)
        self.assertIsNotNone(cached_v)
        
        # Check shapes match
        self.assertEqual(cached_k.shape, self.test_key_tensor.shape)
        self.assertEqual(cached_v.shape, self.test_value_tensor.shape)
        
        # Check values are close (they should be exact for uncompressed cache)
        self.assertTrue(torch.allclose(cached_k, self.test_key_tensor, atol=1e-6))
        self.assertTrue(torch.allclose(cached_v, self.test_value_tensor, atol=1e-6))
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        # Try to get from empty cache
        cached_k, cached_v = self.cache.get("nonexistent_layer")
        
        self.assertIsNone(cached_k)
        self.assertIsNone(cached_v)
    
    def test_cache_update(self):
        """Test updating existing cache entries."""
        layer_name = "layer_0"
        
        # Store initial KV pair
        self.cache.store(layer_name, self.test_key_tensor, self.test_value_tensor)
        
        # Create new tensors
        new_key_tensor = torch.randn(4, 8, 64, device=self.device)
        new_value_tensor = torch.randn(4, 8, 64, device=self.device)
        
        # Update cache
        self.cache.store(layer_name, new_key_tensor, new_value_tensor)
        
        # Retrieve updated values
        cached_k, cached_v = self.cache.get(layer_name)
        
        # Should get new values
        self.assertTrue(torch.allclose(cached_k, new_key_tensor, atol=1e-6))
        self.assertTrue(torch.allclose(cached_v, new_value_tensor, atol=1e-6))
    
    def test_multi_layer_caching(self):
        """Test caching for multiple layers."""
        layers = ["layer_0", "layer_1", "layer_2"]
        tensors = []
        
        # Store different tensors for each layer
        for i, layer in enumerate(layers):
            k_tensor = torch.randn(4, 8, 64, device=self.device) * (i + 1)
            v_tensor = torch.randn(4, 8, 64, device=self.device) * (i + 1)
            tensors.append((k_tensor, v_tensor))
            
            self.cache.store(layer, k_tensor, v_tensor)
        
        # Retrieve and verify each layer
        for i, layer in enumerate(layers):
            cached_k, cached_v = self.cache.get(layer)
            original_k, original_v = tensors[i]
            
            self.assertIsNotNone(cached_k)
            self.assertIsNotNone(cached_v)
            self.assertTrue(torch.allclose(cached_k, original_k, atol=1e-6))
            self.assertTrue(torch.allclose(cached_v, original_v, atol=1e-6))
    
    def test_cache_compression_trigger(self):
        """Test automatic compression when cache size exceeds threshold."""
        # Create small cache to trigger compression quickly
        small_cache_config = self.cache_config.copy()
        small_cache_config['max_cache_size'] = 1000  # Very small cache
        small_cache_config['compression_threshold'] = 0.5
        
        small_cache = ChunkedKVCache(
            config=small_cache_config,
            device=self.device
        )
        
        # Fill cache with large tensors to exceed threshold
        large_tensor = torch.randn(16, 16, 256, device=self.device)
        
        layers = ["layer_0", "layer_1", "layer_2", "layer_3"]
        
        for layer in layers:
            small_cache.store(layer, large_tensor, large_tensor)
        
        # Check that compression was triggered
        stats = small_cache.get_stats()
        
        # Should have some compressed entries
        self.assertGreater(stats.get('compressed_entries', 0), 0)
        
        # Verify we can still retrieve (with some compression loss)
        for layer in layers:
            cached_k, cached_v = small_cache.get(layer)
            self.assertIsNotNone(cached_k)
            self.assertIsNotNone(cached_v)
            self.assertEqual(cached_k.shape, large_tensor.shape)
            self.assertEqual(cached_v.shape, large_tensor.shape)
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        # Perform various cache operations
        layers = ["layer_0", "layer_1"]
        
        for layer in layers:
            self.cache.store(layer, self.test_key_tensor, self.test_value_tensor)
        
        # Get from cache (hits)
        for layer in layers:
            self.cache.get(layer)
        
        # Try to get non-existent (miss)
        self.cache.get("nonexistent_layer")
        
        # Get statistics
        stats = self.cache.get_stats()
        
        # Check required statistics
        required_stats = [
            'total_entries',
            'total_stores',
            'cache_hits',
            'cache_misses',
            'hit_rate',
            'memory_usage_mb',
            'compressed_entries'
        ]
        
        for stat in required_stats:
            self.assertIn(stat, stats)
        
        # Check values are reasonable
        self.assertEqual(stats['total_entries'], len(layers))
        self.assertEqual(stats['total_stores'], len(layers))
        self.assertEqual(stats['cache_hits'], len(layers))
        self.assertEqual(stats['cache_misses'], 1)
        self.assertGreater(stats['memory_usage_mb'], 0)
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Fill cache
        layers = ["layer_0", "layer_1", "layer_2"]
        
        for layer in layers:
            self.cache.store(layer, self.test_key_tensor, self.test_value_tensor)
        
        # Verify cache has entries
        stats_before = self.cache.get_stats()
        self.assertEqual(stats_before['total_entries'], len(layers))
        
        # Clear cache
        self.cache.clear()
        
        # Verify cache is empty
        stats_after = self.cache.get_stats()
        self.assertEqual(stats_after['total_entries'], 0)
        
        # Verify entries cannot be retrieved
        for layer in layers:
            cached_k, cached_v = self.cache.get(layer)
            self.assertIsNone(cached_k)
            self.assertIsNone(cached_v)
    
    def test_selective_cache_clearing(self):
        """Test selective cache clearing by layer pattern."""
        # Fill cache with different layer patterns
        layers = ["encoder_0", "encoder_1", "decoder_0", "decoder_1"]
        
        for layer in layers:
            self.cache.store(layer, self.test_key_tensor, self.test_value_tensor)
        
        # Clear only encoder layers
        self.cache.clear_by_pattern("encoder_*")
        
        # Check encoder layers are cleared
        for layer in ["encoder_0", "encoder_1"]:
            cached_k, cached_v = self.cache.get(layer)
            self.assertIsNone(cached_k)
            self.assertIsNone(cached_v)
        
        # Check decoder layers remain
        for layer in ["decoder_0", "decoder_1"]:
            cached_k, cached_v = self.cache.get(layer)
            self.assertIsNotNone(cached_k)
            self.assertIsNotNone(cached_v)
    
    def test_memory_management(self):
        """Test memory management and size limits."""
        # Create cache with small memory limit
        limited_cache_config = self.cache_config.copy()
        limited_cache_config['max_cache_size'] = 10000  # Small limit
        
        limited_cache = ChunkedKVCache(
            config=limited_cache_config,
            device=self.device
        )
        
        # Fill cache beyond limit
        large_tensor = torch.randn(32, 32, 128, device=self.device)
        
        for i in range(10):  # Try to store many large tensors
            layer_name = f"layer_{i}"
            limited_cache.store(layer_name, large_tensor, large_tensor)
        
        # Check memory usage doesn't exceed limit significantly
        stats = limited_cache.get_stats()
        memory_usage_bytes = stats['memory_usage_mb'] * 1024 * 1024
        
        # Allow some overhead but shouldn't be too much over limit
        self.assertLess(memory_usage_bytes, limited_cache_config['max_cache_size'] * 2)
    
    def test_different_tensor_shapes(self):
        """Test caching with different tensor shapes."""
        shapes = [
            (2, 4, 32),     # Small
            (8, 16, 64),    # Medium
            (4, 8, 128),    # Large sequence
            (16, 4, 32),    # Many heads
            (1, 1, 512)     # Single head, long sequence
        ]
        
        for i, shape in enumerate(shapes):
            with self.subTest(shape=shape):
                layer_name = f"layer_{i}"
                k_tensor = torch.randn(*shape, device=self.device)
                v_tensor = torch.randn(*shape, device=self.device)
                
                # Store and retrieve
                self.cache.store(layer_name, k_tensor, v_tensor)
                cached_k, cached_v = self.cache.get(layer_name)
                
                # Check shapes and values
                self.assertEqual(cached_k.shape, k_tensor.shape)
                self.assertEqual(cached_v.shape, v_tensor.shape)
                self.assertTrue(torch.allclose(cached_k, k_tensor, atol=1e-6))
                self.assertTrue(torch.allclose(cached_v, v_tensor, atol=1e-6))
    
    def test_cache_persistence(self):
        """Test cache persistence to disk."""
        # Fill cache
        layers = ["layer_0", "layer_1"]
        tensors = []
        
        for i, layer in enumerate(layers):
            k_tensor = torch.randn(4, 8, 64, device=self.device)
            v_tensor = torch.randn(4, 8, 64, device=self.device)
            tensors.append((k_tensor, v_tensor))
            
            self.cache.store(layer, k_tensor, v_tensor)
        
        # Save cache
        cache_path = os.path.join(self.temp_dir, 'cache.pt')
        self.cache.save_cache(cache_path)
        
        self.assertTrue(os.path.exists(cache_path))
        
        # Create new cache and load
        new_cache = ChunkedKVCache(
            config=self.cache_config,
            device=self.device
        )
        
        new_cache.load_cache(cache_path)
        
        # Verify loaded cache has same data
        for i, layer in enumerate(layers):
            cached_k, cached_v = new_cache.get(layer)
            original_k, original_v = tensors[i]
            
            self.assertIsNotNone(cached_k)
            self.assertIsNotNone(cached_v)
            self.assertTrue(torch.allclose(cached_k, original_k, atol=1e-5))
            self.assertTrue(torch.allclose(cached_v, original_v, atol=1e-5))
    
    def test_compression_quality(self):
        """Test quality of compressed cache entries."""
        # Create cache with aggressive compression
        aggressive_config = self.cache_config.copy()
        aggressive_config['compression_ratio'] = 0.3
        aggressive_config['max_cache_size'] = 1000  # Force compression
        
        aggressive_cache = ChunkedKVCache(
            config=aggressive_config,
            device=self.device
        )
        
        # Store large tensor that will be compressed
        large_tensor = torch.randn(16, 16, 256, device=self.device)
        layer_name = "test_layer"
        
        aggressive_cache.store(layer_name, large_tensor, large_tensor)
        
        # Force compression by exceeding cache size
        for i in range(5):
            dummy_tensor = torch.randn(16, 16, 256, device=self.device)
            aggressive_cache.store(f"dummy_{i}", dummy_tensor, dummy_tensor)
        
        # Retrieve compressed tensor
        cached_k, cached_v = aggressive_cache.get(layer_name)
        
        if cached_k is not None:  # Might be evicted due to size limits
            # Check approximate reconstruction
            k_error = torch.norm(cached_k - large_tensor) / torch.norm(large_tensor)
            v_error = torch.norm(cached_v - large_tensor) / torch.norm(large_tensor)
            
            # Errors should be reasonable for 30% compression
            self.assertLess(k_error, 0.8)
            self.assertLess(v_error, 0.8)
    
    def test_cache_with_gradients(self):
        """Test cache behavior with tensors that have gradients."""
        # Create tensors with gradients
        k_tensor = torch.randn(4, 8, 64, device=self.device, requires_grad=True)
        v_tensor = torch.randn(4, 8, 64, device=self.device, requires_grad=True)
        
        # Perform some operation to create gradients
        loss = (k_tensor.sum() + v_tensor.sum())
        loss.backward()
        
        # Store in cache
        layer_name = "grad_layer"
        self.cache.store(layer_name, k_tensor, v_tensor)
        
        # Retrieve from cache
        cached_k, cached_v = self.cache.get(layer_name)
        
        # Check retrieval works (gradients should be detached)
        self.assertIsNotNone(cached_k)
        self.assertIsNotNone(cached_v)
        self.assertFalse(cached_k.requires_grad)
        self.assertFalse(cached_v.requires_grad)
    
    def test_cache_thread_safety(self):
        """Test cache thread safety (basic test)."""
        import threading
        import time
        
        # Thread-safe operations counter
        operations_completed = {'count': 0}
        lock = threading.Lock()
        
        def cache_worker(thread_id):
            for i in range(10):
                layer_name = f"thread_{thread_id}_layer_{i}"
                k_tensor = torch.randn(4, 8, 32, device=self.device)
                v_tensor = torch.randn(4, 8, 32, device=self.device)
                
                # Store
                self.cache.store(layer_name, k_tensor, v_tensor)
                
                # Small delay
                time.sleep(0.001)
                
                # Retrieve
                cached_k, cached_v = self.cache.get(layer_name)
                
                # Verify
                if cached_k is not None and cached_v is not None:
                    with lock:
                        operations_completed['count'] += 1
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)
        
        # Check that operations completed without major issues
        self.assertGreater(operations_completed['count'], 20)  # Most operations should succeed


class TestKVCacheIntegration(unittest.TestCase):
    """Integration tests for KV cache with transformer attention."""
    
    def setUp(self):
        """Set up test fixtures with attention mechanism."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create mock attention layer
        class MockAttention(torch.nn.Module):
            def __init__(self, hidden_size=128, num_heads=8):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads
                
                self.query = torch.nn.Linear(hidden_size, hidden_size)
                self.key = torch.nn.Linear(hidden_size, hidden_size)
                self.value = torch.nn.Linear(hidden_size, hidden_size)
                
            def forward(self, x, kv_cache=None):
                batch_size, seq_len, _ = x.shape
                
                q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                if kv_cache is not None:
                    # Try to get from cache
                    cached_k, cached_v = kv_cache.get("attention")
                    
                    if cached_k is not None and cached_v is not None:
                        # Concatenate with cached values
                        k = torch.cat([cached_k, k], dim=2)
                        v = torch.cat([cached_v, v], dim=2)
                    
                    # Store updated cache
                    kv_cache.store("attention", k, v)
                
                # Simple attention computation
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attention_weights = torch.nn.functional.softmax(scores, dim=-1)
                output = torch.matmul(attention_weights, v)
                
                # Reshape output
                output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
                
                return output
        
        self.attention_layer = MockAttention().to(self.device)
        
        self.cache_config = {
            'max_cache_size': 2000000,
            'compression_threshold': 0.8,
            'enable_chunked_cache': True,
            'compression_ratio': 0.6
        }
        
        self.cache = ChunkedKVCache(config=self.cache_config, device=self.device)
    
    def test_attention_with_cache(self):
        """Test attention mechanism with KV cache."""
        batch_size, seq_len, hidden_size = 2, 16, 128
        
        # First forward pass
        x1 = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        output1 = self.attention_layer(x1, self.cache)
        
        self.assertEqual(output1.shape, (batch_size, seq_len, hidden_size))
        
        # Check cache was populated
        cached_k, cached_v = self.cache.get("attention")
        self.assertIsNotNone(cached_k)
        self.assertIsNotNone(cached_v)
        
        # Second forward pass (should use cache)
        x2 = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        output2 = self.attention_layer(x2, self.cache)
        
        self.assertEqual(output2.shape, (batch_size, seq_len, hidden_size))
        
        # Cache should now have longer sequences
        cached_k_2, cached_v_2 = self.cache.get("attention")
        self.assertEqual(cached_k_2.shape[2], seq_len * 2)  # Sequence dimension doubled
        self.assertEqual(cached_v_2.shape[2], seq_len * 2)
    
    def test_incremental_generation(self):
        """Test incremental generation with KV cache."""
        batch_size, hidden_size = 1, 128
        max_length = 20
        
        # Simulate incremental generation
        for step in range(max_length):
            # Single token input for generation
            x = torch.randn(batch_size, 1, hidden_size, device=self.device)
            output = self.attention_layer(x, self.cache)
            
            self.assertEqual(output.shape, (batch_size, 1, hidden_size))
            
            # Check cache grows
            cached_k, cached_v = self.cache.get("attention")
            if cached_k is not None:
                expected_seq_len = step + 1
                self.assertEqual(cached_k.shape[2], expected_seq_len)
                self.assertEqual(cached_v.shape[2], expected_seq_len)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2)
