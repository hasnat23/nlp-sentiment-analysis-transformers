import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SentimentClassifier:
    """Wrapper for transformer-based sentiment classification models."""

    SUPPORTED_MODELS = [
        'bert-base-uncased',
        'bert-large-uncased',
        'roberta-base',
        'roberta-large',
        'distilbert-base-uncased',
        'albert-base-v2',
    ]

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f'Initializing {model_name} with {num_labels} labels on {self.device}')

        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_rate,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

    def tokenize(self, texts, max_length: int = 512, padding: bool = True, truncation: bool = True):
        """Tokenize a list of texts."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt',
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )
        return outputs

    def save(self, output_dir: str):
        """Save model and tokenizer to disk."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f'Model saved to {output_dir}')

    @classmethod
    def load(cls, model_dir: str, num_labels: int = 2, device: Optional[str] = None):
        """Load a saved model from disk."""
        instance = cls.__new__(cls)
        instance.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        instance.num_labels = num_labels
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model.to(instance.device)
        return instance

    def get_model_size(self) -> Dict[str, Any]:
        """Return model parameter counts."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),
        }
