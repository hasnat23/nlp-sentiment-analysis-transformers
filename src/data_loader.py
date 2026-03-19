import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SentimentDataLoader:
    """Load and preprocess datasets for sentiment analysis."""

    SUPPORTED_DATASETS = {
        'imdb': 'imdb',
        'sst2': 'glue',
        'yelp': 'yelp_review_full',
        'amazon': 'amazon_polarity',
    }

    def __init__(self, dataset_name: str = 'imdb', max_samples: Optional[int] = None):
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.dataset = None

    def load_hf_dataset(self) -> DatasetDict:
        """Load dataset from Hugging Face Hub."""
        logger.info(f'Loading dataset: {self.dataset_name}')

        if self.dataset_name == 'sst2':
            self.dataset = load_dataset('glue', 'sst2')
        elif self.dataset_name in self.SUPPORTED_DATASETS:
            self.dataset = load_dataset(self.SUPPORTED_DATASETS[self.dataset_name])
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset_name}')

        return self.dataset

    def load_custom_csv(self, filepath: str, text_col: str = 'text', label_col: str = 'label') -> DatasetDict:
        """Load a custom CSV dataset."""
        logger.info(f'Loading custom CSV from: {filepath}')
        df = pd.read_csv(filepath)
        if self.max_samples:
            df = df.sample(n=min(self.max_samples, len(df)), random_state=42)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df),
            'test': Dataset.from_pandas(test_df),
        })
        return self.dataset

    def get_label_mapping(self) -> Dict[int, str]:
        """Get integer-to-label name mapping."""
        if self.dataset_name == 'yelp':
            return {0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'}
        return {0: 'negative', 1: 'positive'}

    def get_statistics(self) -> Dict:
        """Print dataset statistics."""
        if not self.dataset:
            raise RuntimeError('Dataset not loaded yet.')
        stats = {}
        for split, ds in self.dataset.items():
            stats[split] = {'num_samples': len(ds)}
            logger.info(f'{split}: {len(ds)} samples')
        return stats
