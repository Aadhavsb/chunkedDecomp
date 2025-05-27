# ChunkedDecomp: Efficient KV Cache Compression for Transformers

A production-ready implementation of chunked SVD decomposition for compressing Key-Value (KV) caches in transformer models, enabling efficient inference with reduced memory footprint.

## 🚀 Quick Start

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Clone the repository
cd c:\Users\bhara\repos\chunkedDecomp

# Create conda environment
conda env create -f environment.yml
conda activate chunked-decomp
```

#### Option B: Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Verify Installation
```bash
# Run basic tests
python -m pytest tests/ -v

# Quick functionality check
python -c "from src import ChunkedDecomp; print('✅ Installation successful!')"
```

## 📖 Detailed Setup Guide

### System Requirements
- **Python**: 3.8 or higher
- **PyTorch**: 2.0.0 or higher (with CUDA support recommended)
- **Memory**: 8GB+ RAM (16GB+ recommended for larger models)
- **Storage**: 2GB+ free space for models and data

### Dependencies Installation

The project requires several key packages:

1. **Core ML Libraries**:
   - `torch>=2.0.0` - PyTorch for neural networks
   - `transformers>=4.30.0` - Hugging Face transformers
   - `datasets>=2.10.0` - Dataset loading and processing

2. **Computation & Analysis**:
   - `numpy>=1.24.0` - Numerical computations
   - `scipy>=1.10.0` - Scientific computing (SVD operations)
   - `pandas>=2.0.0` - Data manipulation

3. **Visualization & Monitoring**:
   - `matplotlib>=3.7.0` - Plotting
   - `seaborn>=0.12.0` - Statistical visualizations
   - `wandb>=0.15.0` - Experiment tracking

4. **Utilities**:
   - `accelerate>=0.20.0` - Model acceleration
   - `tqdm>=4.65.0` - Progress bars

### GPU Setup (Optional but Recommended)

If you have NVIDIA GPU:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 How to Use ChunkedDecomp

### Basic Usage Example

```python
from src import ChunkedDecomp, MemoryTracker
from transformers import AutoModel, AutoTokenizer

# 1. Load a transformer model
model_name = "gpt2"  # or any other transformer model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Initialize ChunkedDecomp with different strategies

# Option A: Uniform Strategy (Default) - Same compression for all chunks
compressor_uniform = ChunkedDecomp(
    model_name_or_path=model_name,
    compression_strategy="uniform",
    base_compression_ratio=0.5,  # 50% compression for all chunks
    chunk_size=64
)

# Option B: Progressive Strategy - Early chunks less compressed
compressor_progressive = ChunkedDecomp(
    model_name_or_path=model_name,
    compression_strategy="progressive",
    base_compression_ratio=0.5,  # Average compression
    start_ratio=0.8,              # Early chunks: 80% rank (less compression)
    end_ratio=0.2,                # Later chunks: 20% rank (more compression)
    chunk_size=64
)

# Option C: Adaptive Strategy - Data-driven rank selection
compressor_adaptive = ChunkedDecomp(
    model_name_or_path=model_name,
    compression_strategy="adaptive",
    base_compression_ratio=0.5,
    chunk_size=64
)

# Option D: Custom Manual Ranks - Full control
custom_rank_map = {0: 48, 1: 32, 2: 24, 3: 16}  # Decreasing ranks
compressor_custom = ChunkedDecomp(
    model_name_or_path=model_name,
    decomp_rank_map=custom_rank_map,
    chunk_size=64
)

# 3. Use any compressor (example with progressive)
compressor = compressor_progressive

# 4. Track memory usage during inference
with MemoryTracker() as tracker:
    # Forward pass with compression
    inputs = tokenizer("Hello world", return_tensors="pt")
    result = compressor.forward_with_compression(inputs['input_ids'])
    outputs = result['model_outputs']

print(f"Peak memory usage: {tracker.get_peak_memory():.2f} MB")
print(f"Compression ratio achieved: {result['compression_stats']['effective_compression_ratio']:.3f}")
print(f"Rank map used: {compressor.decomp_rank_map}")
```

### Configuration-Based Usage

Create a configuration file or use existing ones to define different compression strategies:

```python
import yaml
from src import ChunkedDecomp

# Load configuration with multiple strategies
with open('configs/compression_configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Conservative strategy - prioritizes quality
compressor_conservative = ChunkedDecomp(**config['conservative'])
# Example config: uniform strategy, compression_ratio=0.7

# Balanced strategy - good quality/compression trade-off  
compressor_balanced = ChunkedDecomp(**config['balanced'])
# Example config: progressive strategy, start_ratio=0.8, end_ratio=0.3

# Aggressive strategy - maximum compression
compressor_aggressive = ChunkedDecomp(**config['aggressive'])
# Example config: uniform strategy, compression_ratio=0.3

# Or define strategies programmatically:
strategies = {
    'uniform_50': {
        'compression_strategy': 'uniform',
        'base_compression_ratio': 0.5,
        'chunk_size': 64
    },
    'progressive_balanced': {
        'compression_strategy': 'progressive', 
        'base_compression_ratio': 0.5,
        'start_ratio': 0.8,
        'end_ratio': 0.2,
        'chunk_size': 64
    },
    'adaptive_smart': {
        'compression_strategy': 'adaptive',
        'base_compression_ratio': 0.5,
        'chunk_size': 64
    }
}

# Use any strategy
compressor = ChunkedDecomp(
    model_name_or_path="gpt2",
    **strategies['progressive_balanced']
)
```

### Understanding Compression Strategies

ChunkedDecomp offers four main approaches to rank selection:

#### 1. **Uniform Strategy** (Default)
- **What it does**: Uses the same compression rank for all chunks
- **When to use**: Simple compression needs, predictable memory usage
- **Rank calculation**: `rank = chunk_size × compression_ratio`
- **Example**: compression_ratio=0.5, chunk_size=64 → all chunks get rank=32

#### 2. **Progressive Strategy** 
- **What it does**: Higher ranks for early chunks, lower ranks for later chunks
- **When to use**: When early tokens are more important (common in transformers)
- **Rank calculation**: Linear interpolation from `start_ratio` to `end_ratio`
- **Example**: start_ratio=0.8, end_ratio=0.2, chunk_size=64
  - Chunk 0: rank=51 (80% of 64)
  - Chunk 1: rank=38 (60% of 64) 
  - Chunk 2: rank=25 (40% of 64)
  - Chunk 3: rank=13 (20% of 64)

#### 3. **Adaptive Strategy**
- **What it does**: Analyzes actual data to determine optimal ranks per chunk
- **When to use**: Maximum quality preservation, willing to spend calibration time
- **Rank calculation**: Based on SVD energy analysis of representative data
- **Example**: Might assign ranks like [45, 38, 28, 22] based on data importance

#### 4. **Custom Manual Strategy**
- **What it does**: You specify exact rank for each chunk
- **When to use**: Expert tuning, specific domain knowledge
- **Rank calculation**: User-provided mapping
- **Example**: `{0: 48, 1: 32, 2: 24, 3: 16}` for decreasing importance

```python
# Quick comparison of strategies
strategies_demo = {
    'uniform': ChunkedDecomp("gpt2", compression_strategy="uniform", base_compression_ratio=0.5),
    'progressive': ChunkedDecomp("gpt2", compression_strategy="progressive", start_ratio=0.8, end_ratio=0.2),
    'adaptive': ChunkedDecomp("gpt2", compression_strategy="adaptive", base_compression_ratio=0.5),
    'custom': ChunkedDecomp("gpt2", decomp_rank_map={0: 48, 1: 32, 2: 24, 3: 16})
}

for name, compressor in strategies_demo.items():
    print(f"{name}: {compressor.decomp_rank_map}")
```

## 🔧 Running Different Scripts

### 1. Compression Analysis Script

**What it does**: Compresses models and analyzes performance vs memory trade-offs

#### Basic Compression Strategies

```bash
# Uniform Strategy (Default) - Same rank for all chunks
python scripts/run_compression.py \
    --model_name gpt2 \
    --compression_ratio 0.5 \
    --strategy uniform \
    --output_dir results/uniform/

# Progressive Strategy - Higher ranks for early chunks, lower for later chunks
python scripts/run_compression.py \
    --model_name gpt2 \
    --compression_ratio 0.5 \
    --strategy progressive \
    --start_ratio 0.8 \
    --end_ratio 0.2 \
    --output_dir results/progressive/

# Adaptive Strategy - Data-driven rank selection
python scripts/run_compression.py \
    --model_name gpt2 \
    --compression_ratio 0.5 \
    --strategy adaptive \
    --dataset_name wikitext \
    --calibration_samples 500 \
    --output_dir results/adaptive/
```

#### Comprehensive Strategy Comparison

```bash
# Compare all strategies with multiple compression ratios
python scripts/run_compression.py \
    --model_name microsoft/DialoGPT-medium \
    --compression_ratio 0.3 0.5 0.7 \
    --strategy uniform progressive adaptive \
    --chunk_size 32 64 128 \
    --dataset_name wikitext \
    --num_samples 1000 \
    --comparison_study \
    --output_dir results/strategy_comparison/

# Custom rank mapping example
python scripts/run_compression.py \
    --model_name gpt2 \
    --custom_ranks "0:48,1:32,2:24,3:16" \
    --dataset_name wikitext \
    --output_dir results/custom_ranks/
```

**Strategy Explanations**:
- **Uniform**: `rank = chunk_size × compression_ratio` for all chunks
- **Progressive**: Early chunks get `start_ratio`, later chunks get `end_ratio`
- **Adaptive**: Analyzes data importance to determine optimal ranks per chunk
- **Custom**: Manually specify exact rank for each chunk

**Output**: Compression statistics, performance metrics, memory usage plots, strategy comparison charts

### 2. Model Evaluation Script

**What it does**: Evaluates model quality before/after compression

```bash
# Evaluate compressed model quality
python scripts/evaluate_model.py \
    --model_name gpt2 \
    --compression_ratio 0.5 \
    --metrics perplexity bleu rouge \
    --dataset_name wikitext \
    --output_dir results/evaluation/

# Compare multiple compression settings
python scripts/evaluate_model.py \
    --model_name gpt2 \
    --compression_ratio 0.2 0.4 0.6 0.8 \
    --comparison_study \
    --save_plots
```

**Output**: Perplexity scores, BLEU/ROUGE metrics, quality vs compression plots

### 3. Memory Benchmarking Script

**What it does**: Analyzes memory usage patterns and optimization

```bash
# Memory scaling analysis
python scripts/benchmark_memory.py \
    --model_name gpt2 \
    --sequence_lengths 128 256 512 1024 \
    --batch_sizes 1 4 8 16 \
    --compression_ratios 0.3 0.5 0.7 \
    --output_dir results/memory_analysis/

# Memory profiling for specific model
python scripts/benchmark_memory.py \
    --model_name microsoft/DialoGPT-small \
    --profile_compression \
    --save_timeline
```

**Output**: Memory usage timelines, scaling analysis, optimization recommendations

## 📊 Interactive Analysis (Jupyter Notebook)

**Best for**: Exploratory analysis, parameter tuning, visualization

```bash
# Start Jupyter
jupyter notebook

# Open the exploration notebook
# Navigate to: notebooks/chunked_decomp_exploration.ipynb
```

**The notebook includes**:
- SVD compression exploration
- Interactive parameter tuning
- Real-time memory monitoring
- Model comparison tools
- Visualization dashboards

## 🎛️ Using Different Models

### Supported Model Types

The system works with any Hugging Face transformer model:

```python
# Language Models
models = [
    "gpt2",                          # Small GPT-2 (124M params)
    "gpt2-medium",                   # Medium GPT-2 (355M params)
    "microsoft/DialoGPT-small",      # Conversational model
    "distilgpt2",                    # Distilled GPT-2
    "openai-gpt",                    # Original GPT
]

# BERT-style models
bert_models = [
    "bert-base-uncased",
    "distilbert-base-uncased", 
    "roberta-base",
]

# Larger models (requires more memory)
large_models = [
    "gpt2-large",                    # Large GPT-2 (774M params)
    "gpt2-xl",                       # XL GPT-2 (1.5B params)
    "microsoft/DialoGPT-large",      # Large conversational model
]
```

### Model-Specific Configuration

```python
# For smaller models (< 500M params)
small_model_config = {
    'compression_ratio': 0.6,
    'chunk_size': 64,
    'adaptive_rank': True,
    'target_rank': None  # Auto-determined
}

# For larger models (> 500M params)
large_model_config = {
    'compression_ratio': 0.4,  # More aggressive compression
    'chunk_size': 128,         # Larger chunks for efficiency
    'adaptive_rank': True,
    'min_rank_ratio': 0.1      # Minimum rank preservation
}

# Use with specific model
from src import ChunkedDecomp

compressor = ChunkedDecomp(**large_model_config)
compressed_model = compressor.compress_model_from_name("gpt2-large")
```

## 🔍 Checking If It Works

### 1. Quick Functionality Test

```python
# Test basic compression
from src import ChunkedDecomp
from transformers import AutoModel

model = AutoModel.from_pretrained("distilgpt2")
compressor = ChunkedDecomp(compression_ratio=0.5)

try:
    compressed = compressor.compress_model(model)
    print("✅ Compression successful!")
    
    # Check compression ratio
    original_size = sum(p.numel() for p in model.parameters())
    compressed_size = sum(p.numel() for p in compressed.parameters())
    ratio = compressed_size / original_size
    print(f"✅ Actual compression ratio: {ratio:.3f}")
    
except Exception as e:
    print(f"❌ Error: {e}")
```

### 2. End-to-End Pipeline Test

```bash
# Run mini test pipeline
python scripts/run_compression.py \
    --model_name distilgpt2 \
    --compression_ratio 0.5 \
    --num_samples 100 \
    --test_mode \
    --output_dir results/test/
```

### 3. Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_chunked_decomp.py -v
python -m pytest tests/test_compression.py -v
python -m pytest tests/test_kv_cache.py -v
```

## 📈 Checking How Well It Works

### 1. Performance Metrics

**Compression Effectiveness**:
```python
from src.evaluation import PerformanceEvaluator

evaluator = PerformanceEvaluator()
metrics = evaluator.evaluate_compression(
    original_model=original_model,
    compressed_model=compressed_model,
    test_data=test_dataset
)

print(f"Compression ratio: {metrics['compression_ratio']:.3f}")
print(f"Perplexity increase: {metrics['perplexity_increase']:.2f}%")
print(f"Memory reduction: {metrics['memory_reduction']:.2f}%")
print(f"Speed improvement: {metrics['speed_improvement']:.2f}x")
```

**Quality Metrics**:
```python
# Language quality assessment
quality_metrics = evaluator.evaluate_quality(
    model=compressed_model,
    reference_texts=reference_texts,
    generated_texts=generated_texts
)

print(f"BLEU score: {quality_metrics['bleu']:.3f}")
print(f"ROUGE-L: {quality_metrics['rouge_l']:.3f}")
print(f"Semantic similarity: {quality_metrics['semantic_sim']:.3f}")
```

### 2. Memory Analysis

```python
from src.evaluation import MemoryProfiler

profiler = MemoryProfiler()

# Profile memory usage during inference
with profiler.profile_inference():
    outputs = model(input_ids)

# Get detailed memory breakdown
memory_report = profiler.generate_report()
print(memory_report)

# Visualize memory timeline
profiler.plot_memory_timeline(save_path="memory_analysis.png")
```

### 3. Comparative Analysis

```bash
# Compare multiple compression ratios
python scripts/evaluate_model.py \
    --model_name gpt2 \
    --compression_ratio 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --comparison_study \
    --save_plots \
    --output_dir results/comparison/
```

This generates:
- Performance vs compression trade-off plots
- Memory usage comparisons
- Quality degradation analysis
- Optimal compression recommendations

### 4. Benchmark Against Baselines

```python
from src.utils import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()

# Run comprehensive comparison
results = benchmark.run_scaling_analysis(
    model_name="gpt2",
    compression_ratios=[0.3, 0.5, 0.7],
    sequence_lengths=[128, 256, 512],
    batch_sizes=[1, 4, 8]
)

# Generate recommendation
recommendation = benchmark.recommend_optimal_settings(
    target_memory_reduction=0.4,  # 40% memory reduction target
    max_quality_loss=0.1          # Max 10% quality loss
)

print(f"Recommended compression ratio: {recommendation['compression_ratio']}")
print(f"Expected memory reduction: {recommendation['memory_reduction']:.1%}")
print(f"Expected quality loss: {recommendation['quality_loss']:.1%}")
```

## 🎛️ Advanced Configuration

### Custom Compression Strategies

```python
# Conservative compression (high quality)
conservative_config = {
    'compression_ratio': 0.7,
    'adaptive_rank': True,
    'min_rank_ratio': 0.3,
    'chunk_size': 32,
    'use_low_rank_approximation': True
}

# Aggressive compression (maximum memory savings)
aggressive_config = {
    'compression_ratio': 0.3,
    'adaptive_rank': True,
    'min_rank_ratio': 0.05,
    'chunk_size': 128,
    'use_quantization': True
}

# Balanced compression (good trade-off)
balanced_config = {
    'compression_ratio': 0.5,
    'adaptive_rank': True,
    'min_rank_ratio': 0.15,
    'chunk_size': 64,
    'dynamic_chunking': True
}
```

### Environment Variables

```bash
# Set for better performance
export TORCH_BACKENDS_CUDNN_BENCHMARK=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# For debugging
export CHUNKED_DECOMP_DEBUG=1
export CHUNKED_DECOMP_LOG_LEVEL=INFO
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
```python
# Reduce batch size or sequence length
compressor = ChunkedDecomp(
    compression_ratio=0.4,  # More aggressive compression
    chunk_size=32,          # Smaller chunks
    enable_gradient_checkpointing=True
)
```

2. **Poor Compression Quality**:
```python
# Use more conservative settings
compressor = ChunkedDecomp(
    compression_ratio=0.7,  # Less aggressive
    adaptive_rank=True,
    min_rank_ratio=0.2      # Preserve more information
)
```

3. **Slow Performance**:
```python
# Optimize for speed
compressor = ChunkedDecomp(
    chunk_size=128,         # Larger chunks
    parallel_processing=True,
    use_fast_svd=True
)
```

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode
compressor = ChunkedDecomp(debug=True)
```

## 📚 Next Steps

1. **Start with the Jupyter notebook** for interactive exploration
2. **Run small tests** with `distilgpt2` model first
3. **Scale up gradually** to larger models
4. **Experiment with different configurations** based on your use case
5. **Monitor performance metrics** to find optimal settings

## 🆘 Getting Help

- Check the test files for usage examples
- Use the Jupyter notebook for interactive learning  
- Enable debug mode for detailed error information
- Run with smaller models first to verify setup

Happy compressing! 🚀
