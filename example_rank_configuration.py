#!/usr/bin/env python3
"""
Example: Understanding and Configuring Compression Ranks in ChunkedDecomp
"""

import torch
from src import ChunkedDecomp
from src.utils.svd_utils import create_adaptive_rank_map, SVDCompressor

def main():
    print("🎯 Understanding Compression Rank in ChunkedDecomp")
    print("=" * 60)
    
    # =================================================================
    # 1. MANUAL RANK CONFIGURATION
    # =================================================================
    print("\n1. Manual Rank Configuration")
    print("-" * 30)
    
    # Create custom rank mapping
    chunk_size = 64
    n_chunks = 4  # For head_dim = 256
    
    # Manual rank mapping - different ranks per chunk
    custom_rank_map = {
        0: 32,  # Chunk 0: 50% compression
        1: 16,  # Chunk 1: 25% compression (more aggressive)
        2: 48,  # Chunk 2: 75% compression (less aggressive)
        3: 24   # Chunk 3: 37.5% compression
    }
    
    print(f"Custom rank map: {custom_rank_map}")
    
    # Initialize with custom ranks
    compressor = ChunkedDecomp(
        model_name_or_path="gpt2",
        chunk_size=chunk_size,
        decomp_rank_map=custom_rank_map  # Use custom ranks
    )
    
    print(f"✅ Initialized with custom ranks")
    
    # =================================================================
    # 2. STRATEGY-BASED RANK CONFIGURATION
    # =================================================================
    print("\n2. Strategy-Based Rank Configuration")
    print("-" * 40)
    
    strategies = ["uniform", "progressive", "adaptive"]
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} Strategy ---")
        
        rank_map = create_adaptive_rank_map(
            n_chunks=4,
            chunk_size=64,
            strategy=strategy,
            base_compression_ratio=0.5,
            start_ratio=0.8,  # For progressive
            end_ratio=0.2     # For progressive
        )
        
        print(f"Rank map: {rank_map}")
        
        # Calculate compression ratios for each chunk
        for chunk_idx, rank in rank_map.items():
            ratio = rank / chunk_size
            print(f"  Chunk {chunk_idx}: rank={rank}, ratio={ratio:.2f}")
    
    # =================================================================
    # 3. ADAPTIVE RANK SELECTION BASED ON DATA
    # =================================================================
    print("\n3. Adaptive Rank Selection Based on Data")
    print("-" * 42)
    
    # Create SVD compressor for analysis
    svd_compressor = SVDCompressor()
    
    # Simulate different types of data
    test_matrices = {
        "Low-rank": torch.randn(100, 64) @ torch.randn(64, 10) @ torch.randn(10, 64),
        "Full-rank": torch.randn(100, 64),
        "Sparse": torch.zeros(100, 64).fill_diagonal_(1.0)
    }
    
    for name, matrix in test_matrices.items():
        print(f"\n--- {name} Matrix ---")
        
        # Analyze optimal rank for different compression targets
        targets = [0.3, 0.5, 0.7]
        
        for target in targets:
            analysis = svd_compressor.optimal_rank_analysis(
                matrix, 
                target_compression_ratio=target
            )
            
            print(f"Target ratio {target}: "
                  f"rank={analysis['recommended_rank']}, "
                  f"actual={analysis['actual_compression_ratio']:.3f}, "
                  f"error={analysis['relative_truncation_error']:.4f}")
    
    # =================================================================
    # 4. DYNAMIC RANK ADJUSTMENT DURING RUNTIME
    # =================================================================
    print("\n4. Dynamic Rank Adjustment")
    print("-" * 28)
    
    # Create compressor with adaptive rank enabled
    adaptive_compressor = ChunkedDecomp(
        model_name_or_path="gpt2",
        compression_strategy="adaptive",
        base_compression_ratio=0.5
    )
    
    print("✅ Adaptive compressor initialized")
    print(f"Rank map: {adaptive_compressor.decomp_rank_map}")
    
    # =================================================================
    # 5. COMPRESSION RATIO vs RANK RELATIONSHIP
    # =================================================================
    print("\n5. Compression Ratio vs Rank Relationship")
    print("-" * 43)
    
    chunk_size = 64
    seq_len = 100
    
    print(f"For matrix size ({seq_len} × {chunk_size}):")
    print("Rank\tCompression Ratio\tMemory Usage")
    print("-" * 45)
    
    original_elements = seq_len * chunk_size
    
    for rank in [8, 16, 32, 48, 56]:
        # Compressed storage: A(seq_len × rank) + B(chunk_size × rank)
        compressed_elements = seq_len * rank + chunk_size * rank
        compression_ratio = compressed_elements / original_elements
        
        print(f"{rank}\t{compression_ratio:.3f}\t\t{compressed_elements}/{original_elements}")
    
    # =================================================================
    # 6. QUALITY ANALYSIS BY RANK
    # =================================================================
    print("\n6. Quality Analysis by Rank")
    print("-" * 27)
    
    test_matrix = torch.randn(50, 64)
    
    print("Rank\tReconstruction Error\tEnergy Preserved")
    print("-" * 48)
    
    for rank in [8, 16, 32, 48]:
        # Compress with specific rank
        compressed_data = svd_compressor.compress_matrix(
            test_matrix,
            rank=rank
        )
        
        # Decompress and compute error
        reconstructed = svd_compressor.decompress_matrix(compressed_data)
        error = torch.norm(test_matrix - reconstructed) / torch.norm(test_matrix)
        
        print(f"{rank}\t{error:.4f}\t\t{compressed_data.get('rank_energy_ratio', 'N/A')}")
    
    # =================================================================
    # 7. CONFIGURATION FILES EXAMPLE
    # =================================================================
    print("\n7. Configuration File Usage")
    print("-" * 30)
    
    config_example = {
        'compression_strategies': {
            'conservative': {
                'rank_strategy': 'uniform',
                'base_compression_ratio': 0.7,
                'description': 'High quality, less compression'
            },
            'balanced': {
                'rank_strategy': 'progressive', 
                'base_compression_ratio': 0.5,
                'start_ratio': 0.8,
                'end_ratio': 0.3,
                'description': 'Balanced quality/compression'
            },
            'aggressive': {
                'rank_strategy': 'uniform',
                'base_compression_ratio': 0.3,
                'description': 'Maximum compression'
            }
        }
    }
    
    print("Example configuration strategies:")
    for name, config in config_example['compression_strategies'].items():
        print(f"  {name}: ratio={config['base_compression_ratio']}, "
              f"strategy={config['rank_strategy']}")
        print(f"    {config['description']}")
    
    print("\n🎉 Rank configuration examples completed!")
    print("\nKey Takeaways:")
    print("• Lower rank = more compression, potential quality loss")
    print("• Higher rank = better quality, less compression")  
    print("• Use 'adaptive' strategy for automatic rank selection")
    print("• Progressive strategy: higher ranks for early chunks")
    print("• Monitor reconstruction error to validate quality")

if __name__ == "__main__":
    main()
