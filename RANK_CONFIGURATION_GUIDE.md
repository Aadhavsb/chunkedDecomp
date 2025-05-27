# Rank Configuration Guide for ChunkedDecomp

## Quick Reference: Compression Rank Settings

### 1. How Compression Rank Works

**Compression Rank** = Number of singular values kept during SVD compression

```
Original Matrix: (sequence_length × chunk_size)
Compressed Form: A(sequence_length × rank) + B(chunk_size × rank)

Compression Ratio = (seq_len × rank + chunk_size × rank) / (seq_len × chunk_size)
```

### 2. Configuration Methods

#### Method 1: Using Compression Strategies (Recommended)

```python
from src import ChunkedDecomp

# Conservative (high quality, ~70% original size)
compressor = ChunkedDecomp(
    model_name_or_path="gpt2",
    compression_strategy="uniform",
    base_compression_ratio=0.7
)

# Balanced (good trade-off, ~50% original size) 
compressor = ChunkedDecomp(
    model_name_or_path="gpt2", 
    compression_strategy="progressive",
    base_compression_ratio=0.5,
    start_ratio=0.8,  # Early chunks: 80% of chunk_size
    end_ratio=0.3     # Later chunks: 30% of chunk_size
)

# Aggressive (maximum compression, ~30% original size)
compressor = ChunkedDecomp(
    model_name_or_path="gpt2",
    compression_strategy="uniform", 
    base_compression_ratio=0.3
)
```

#### Method 2: Manual Rank Mapping

```python
# Custom rank for each chunk (head_dim = 256, chunk_size = 64)
custom_ranks = {
    0: 48,  # Chunk 0: 75% rank (less compression)
    1: 32,  # Chunk 1: 50% rank (medium compression) 
    2: 16,  # Chunk 2: 25% rank (more compression)
    3: 24   # Chunk 3: 37.5% rank
}

compressor = ChunkedDecomp(
    model_name_or_path="gpt2",
    chunk_size=64,
    decomp_rank_map=custom_ranks
)
```

#### Method 3: Adaptive Rank Selection

```python
# Let the system choose ranks automatically
compressor = ChunkedDecomp(
    model_name_or_path="gpt2",
    compression_strategy="adaptive",
    base_compression_ratio=0.5
)
```

### 3. Understanding Rank Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `uniform` | Same rank for all chunks | Simple, predictable compression |
| `progressive` | Higher ranks for early chunks | When early attention is more important |
| `adaptive` | Data-driven rank selection | Best quality/compression trade-off |

### 4. Compression Ratio Examples

For chunk_size = 64:

| Rank | Compression Ratio | Memory Usage | Quality |
|------|------------------|--------------|---------|
| 8    | ~0.25            | 25% original | Lower quality |
| 16   | ~0.50            | 50% original | Good quality |
| 32   | ~0.75            | 75% original | High quality |
| 48   | ~0.875           | 87.5% original | Excellent quality |

### 5. Dynamic Configuration

```python
# Configuration based on model size
def get_compression_config(model_size_params):
    if model_size_params < 100e6:  # < 100M parameters
        return {
            'compression_strategy': 'uniform',
            'base_compression_ratio': 0.6,
            'chunk_size': 32
        }
    elif model_size_params < 1e9:  # < 1B parameters  
        return {
            'compression_strategy': 'progressive',
            'base_compression_ratio': 0.5,
            'chunk_size': 64,
            'start_ratio': 0.8,
            'end_ratio': 0.3
        }
    else:  # Large models
        return {
            'compression_strategy': 'adaptive',
            'base_compression_ratio': 0.4,
            'chunk_size': 128
        }

# Usage
model_params = 117e6  # GPT2 size
config = get_compression_config(model_params)
compressor = ChunkedDecomp(model_name_or_path="gpt2", **config)
```

### 6. Quality Monitoring

```python
# Check compression quality
from src.utils.svd_utils import SVDCompressor

compressor = SVDCompressor()

# Analyze optimal rank for target compression
analysis = compressor.optimal_rank_analysis(
    your_tensor,
    target_compression_ratio=0.5
)

print(f"Recommended rank: {analysis['recommended_rank']}")
print(f"Expected error: {analysis['relative_truncation_error']:.4f}")
```

### 7. Configuration Files

See `configs/compression_configs.yaml` for pre-configured strategies:

```yaml
compression_strategies:
  conservative:
    rank_strategy: "uniform"
    base_compression_ratio: 0.7
    
  balanced:
    rank_strategy: "progressive" 
    base_compression_ratio: 0.5
    start_ratio: 0.8
    end_ratio: 0.3
    
  aggressive:
    rank_strategy: "uniform"
    base_compression_ratio: 0.3
```

### 8. Best Practices

1. **Start Conservative**: Begin with `base_compression_ratio=0.6` and adjust based on quality metrics
2. **Use Progressive for Attention**: Early attention heads often more important
3. **Monitor Quality**: Check perplexity and reconstruction error
4. **Chunk Size Trade-off**: Larger chunks = better compression efficiency, but less granular control
5. **Model-Specific Tuning**: Different models may need different rank strategies

### 9. Troubleshooting

**Quality too low?**
- Increase `base_compression_ratio` (e.g., 0.3 → 0.5)
- Use `progressive` strategy with higher `start_ratio`
- Increase `chunk_size` for better granularity

**Not enough compression?**
- Decrease `base_compression_ratio` (e.g., 0.7 → 0.4) 
- Use `uniform` strategy for consistent compression
- Consider `aggressive` preset configuration

**Out of memory?**
- Increase compression (lower `base_compression_ratio`)
- Use smaller `chunk_size`
- Enable gradient checkpointing if available
