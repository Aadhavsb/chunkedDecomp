"""Data utilities for loading and preprocessing datasets for chunked decomposition."""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from pathlib import Path
import json
import pickle
from datasets import load_dataset, Dataset as HFDataset
import random

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text data with tokenization."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True,
        return_attention_mask: bool = True
    ):
        """Initialize text dataset.
        
        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            return_attention_mask: Whether to return attention masks
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_attention_mask = return_attention_mask
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors='pt',
            return_attention_mask=self.return_attention_mask
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add labels for language modeling (shifted input_ids)
        if 'input_ids' in item:
            item['labels'] = item['input_ids'].clone()
        
        return item


class BenchmarkDataset(Dataset):
    """Dataset specifically designed for benchmarking models."""
    
    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        vocab_size: int = 32000,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ):
        """Initialize benchmark dataset with synthetic data.
        
        Args:
            num_samples: Number of samples to generate
            seq_length: Sequence length for each sample
            vocab_size: Vocabulary size for token generation
            batch_size: Batch size (for pre-batching)
            device: Device to place tensors on
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
        
        # Pre-generate data for consistent benchmarking
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic data for benchmarking."""
        data = []
        
        for i in range(self.num_samples):
            # Generate random token IDs
            input_ids = torch.randint(
                0, self.vocab_size, 
                (self.seq_length,), 
                device=self.device
            )
            
            # Create attention mask (all 1s for synthetic data)
            attention_mask = torch.ones_like(input_ids, device=self.device)
            
            # Labels are the same as input_ids for language modeling
            labels = input_ids.clone()
            
            item = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
            data.append(item)
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class DatasetManager:
    """Manager for loading and preprocessing various datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize dataset manager.
        
        Args:
            cache_dir: Directory to cache processed datasets
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / 'data' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_wikitext(
        self,
        version: str = 'wikitext-2-raw-v1',
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        cache: bool = True
    ) -> Dict[str, DataLoader]:
        """Load WikiText dataset.
        
        Args:
            version: WikiText version to load
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            cache: Whether to cache processed data
            
        Returns:
            Dictionary with train/validation/test dataloaders
        """
        cache_file = self.cache_dir / f"wikitext_{version}_{tokenizer_name}_{max_length}.pkl"
        
        if cache and cache_file.exists():
            logger.info(f"Loading cached WikiText dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Loading WikiText dataset: {version}")
        
        # Load dataset from HuggingFace
        dataset = load_dataset("wikitext", version)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Process splits
        dataloaders = {}
        
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                texts = [item['text'] for item in dataset[split] if item['text'].strip()]
                
                # Filter out very short texts
                texts = [text for text in texts if len(text.split()) > 10]
                
                # Create dataset
                text_dataset = TextDataset(
                    texts=texts,
                    tokenizer=tokenizer,
                    max_length=max_length
                )
                
                # Create dataloader
                dataloaders[split] = DataLoader(
                    text_dataset,
                    batch_size=8,
                    shuffle=(split == 'train'),
                    num_workers=0,  # Set to 0 for Windows compatibility
                    pin_memory=torch.cuda.is_available()
                )
        
        # Cache results
        if cache:
            with open(cache_file, 'wb') as f:
                pickle.dump(dataloaders, f)
            logger.info(f"Cached processed dataset to {cache_file}")
        
        return dataloaders
    
    def load_openwebtext(
        self,
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        num_samples: Optional[int] = None,
        cache: bool = True
    ) -> Dict[str, DataLoader]:
        """Load OpenWebText dataset.
        
        Args:
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            num_samples: Limit number of samples (for testing)
            cache: Whether to cache processed data
            
        Returns:
            Dictionary with train/validation dataloaders
        """
        cache_key = f"openwebtext_{tokenizer_name}_{max_length}_{num_samples}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache and cache_file.exists():
            logger.info(f"Loading cached OpenWebText dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info("Loading OpenWebText dataset")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("openwebtext", split='train')
            
            if num_samples:
                # Take a subset for testing
                indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
                dataset = dataset.select(indices)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Extract texts
            texts = [item['text'] for item in dataset if item['text'].strip()]
            
            # Filter out very short texts
            texts = [text for text in texts if len(text.split()) > 20]
            
            # Split into train/validation
            split_idx = int(len(texts) * 0.9)
            train_texts = texts[:split_idx]
            val_texts = texts[split_idx:]
            
            dataloaders = {}
            
            # Create train dataset
            train_dataset = TextDataset(
                texts=train_texts,
                tokenizer=tokenizer,
                max_length=max_length
            )
            dataloaders['train'] = DataLoader(
                train_dataset,
                batch_size=8,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            # Create validation dataset
            val_dataset = TextDataset(
                texts=val_texts,
                tokenizer=tokenizer,
                max_length=max_length
            )
            dataloaders['validation'] = DataLoader(
                val_dataset,
                batch_size=8,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            # Cache results
            if cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(dataloaders, f)
                logger.info(f"Cached processed dataset to {cache_file}")
            
            return dataloaders
            
        except Exception as e:
            logger.warning(f"Could not load OpenWebText: {e}")
            logger.info("Falling back to synthetic data")
            return self.create_synthetic_dataset(tokenizer_name, max_length)
    
    def create_synthetic_dataset(
        self,
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        num_train_samples: int = 1000,
        num_val_samples: int = 100,
        batch_size: int = 8
    ) -> Dict[str, DataLoader]:
        """Create synthetic dataset for testing.
        
        Args:
            tokenizer_name: Name of tokenizer to use for vocab size
            max_length: Maximum sequence length
            num_train_samples: Number of training samples
            num_val_samples: Number of validation samples
            batch_size: Batch size for dataloaders
            
        Returns:
            Dictionary with train/validation dataloaders
        """
        logger.info("Creating synthetic dataset")
        
        # Get vocabulary size from tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = tokenizer.vocab_size
        
        # Create datasets
        train_dataset = BenchmarkDataset(
            num_samples=num_train_samples,
            seq_length=max_length,
            vocab_size=vocab_size
        )
        
        val_dataset = BenchmarkDataset(
            num_samples=num_val_samples,
            seq_length=max_length,
            vocab_size=vocab_size
        )
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            ),
            'validation': DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
        }
        
        return dataloaders
    
    def create_benchmark_dataloader(
        self,
        batch_size: int = 1,
        seq_length: int = 512,
        num_samples: int = 100,
        vocab_size: int = 32000,
        device: Optional[torch.device] = None
    ) -> DataLoader:
        """Create a dataloader specifically for benchmarking.
        
        Args:
            batch_size: Batch size
            seq_length: Sequence length
            num_samples: Number of samples
            vocab_size: Vocabulary size
            device: Device to place data on
            
        Returns:
            DataLoader for benchmarking
        """
        dataset = BenchmarkDataset(
            num_samples=num_samples,
            seq_length=seq_length,
            vocab_size=vocab_size,
            device=device
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
    
    def load_custom_dataset(
        self,
        data_path: str,
        tokenizer_name: str = 'gpt2',
        max_length: int = 512,
        text_column: str = 'text',
        batch_size: int = 8
    ) -> DataLoader:
        """Load custom dataset from file.
        
        Args:
            data_path: Path to data file (JSON, CSV, or text)
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            text_column: Column name containing text data
            batch_size: Batch size for dataloader
            
        Returns:
            DataLoader for custom dataset
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data based on file extension
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                texts = [item[text_column] if isinstance(item, dict) else str(item) for item in data]
            elif isinstance(data, dict):
                texts = list(data.values())
            else:
                raise ValueError("Unsupported JSON structure")
        
        elif data_path.suffix == '.txt':
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        elif data_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(data_path)
            texts = df[text_column].tolist()
        
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        # Filter out empty texts
        texts = [text for text in texts if text and len(text.strip()) > 0]
        
        logger.info(f"Loaded {len(texts)} texts from {data_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset
        dataset = TextDataset(
            texts=texts,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )


def create_dataloader_from_config(config: Dict[str, Any]) -> DataLoader:
    """Create dataloader from configuration dictionary.
    
    Args:
        config: Configuration dictionary with dataset parameters
        
    Returns:
        Configured dataloader
    """
    dataset_manager = DatasetManager(config.get('cache_dir'))
    
    dataset_type = config.get('type', 'synthetic')
    
    if dataset_type == 'wikitext':
        dataloaders = dataset_manager.load_wikitext(
            version=config.get('version', 'wikitext-2-raw-v1'),
            tokenizer_name=config.get('tokenizer', 'gpt2'),
            max_length=config.get('max_length', 512)
        )
        return dataloaders[config.get('split', 'validation')]
    
    elif dataset_type == 'openwebtext':
        dataloaders = dataset_manager.load_openwebtext(
            tokenizer_name=config.get('tokenizer', 'gpt2'),
            max_length=config.get('max_length', 512),
            num_samples=config.get('num_samples')
        )
        return dataloaders[config.get('split', 'validation')]
    
    elif dataset_type == 'synthetic':
        dataloaders = dataset_manager.create_synthetic_dataset(
            tokenizer_name=config.get('tokenizer', 'gpt2'),
            max_length=config.get('max_length', 512),
            num_train_samples=config.get('num_train_samples', 1000),
            num_val_samples=config.get('num_val_samples', 100),
            batch_size=config.get('batch_size', 8)
        )
        return dataloaders[config.get('split', 'validation')]
    
    elif dataset_type == 'benchmark':
        return dataset_manager.create_benchmark_dataloader(
            batch_size=config.get('batch_size', 1),
            seq_length=config.get('seq_length', 512),
            num_samples=config.get('num_samples', 100),
            vocab_size=config.get('vocab_size', 32000)
        )
    
    elif dataset_type == 'custom':
        return dataset_manager.load_custom_dataset(
            data_path=config['data_path'],
            tokenizer_name=config.get('tokenizer', 'gpt2'),
            max_length=config.get('max_length', 512),
            text_column=config.get('text_column', 'text'),
            batch_size=config.get('batch_size', 8)
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_info(dataloader: DataLoader) -> Dict[str, Any]:
    """Get information about a dataset from its dataloader.
    
    Args:
        dataloader: DataLoader to analyze
        
    Returns:
        Dictionary with dataset information
    """
    dataset = dataloader.dataset
    
    info = {
        'num_samples': len(dataset),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader)
    }
    
    # Get sample to analyze structure
    sample = dataset[0]
    
    if isinstance(sample, dict):
        info['sample_keys'] = list(sample.keys())
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                info[f'{key}_shape'] = list(value.shape)
                info[f'{key}_dtype'] = str(value.dtype)
    
    # Estimate memory usage
    total_elements = info['num_samples']
    if 'input_ids_shape' in info:
        total_elements *= np.prod(info['input_ids_shape'])
    
    # Rough estimate: 4 bytes per token (int32)
    estimated_memory_mb = (total_elements * 4) / (1024 * 1024)
    info['estimated_memory_mb'] = estimated_memory_mb
    
    return info


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/validation/test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Shuffle indices
    random.seed(random_seed)
    random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def collate_fn_variable_length(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable length sequences.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors with padding
    """
    # Get all keys from first sample
    keys = batch[0].keys()
    
    collated = {}
    
    for key in keys:
        tensors = [item[key] for item in batch]
        
        if key in ['input_ids', 'labels']:
            # Pad sequences to max length in batch
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )
        elif key == 'attention_mask':
            # Pad attention masks with 0
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=0
            )
        else:
            # Stack other tensors
            collated[key] = torch.stack(tensors)
    
    return collated
