import os
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
import mlflow
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data_loader import SentimentDataLoader
from model import SentimentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Run one training epoch."""
    model.model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test set."""
    model.model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, f1


def train(
    model_name: str = 'bert-base-uncased',
    dataset_name: str = 'imdb',
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    output_dir: str = './outputs/model',
    num_labels: int = 2,
    warmup_ratio: float = 0.1,
    experiment_name: str = 'sentiment-analysis',
):
    """Full training pipeline with MLflow tracking."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            'model_name': model_name,
            'dataset_name': dataset_name,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_length': max_length,
        })

        # Load data
        logger.info('Loading dataset...')
        loader = SentimentDataLoader(dataset_name=dataset_name)
        dataset = loader.load_hf_dataset()

        # Initialize model
        logger.info(f'Initializing model: {model_name}')
        classifier = SentimentClassifier(
            model_name=model_name,
            num_labels=num_labels,
        )

        # Tokenize data
        def tokenize_fn(examples):
            return classifier.tokenizer(
                examples['text'],
                max_length=max_length,
                padding='max_length',
                truncation=True,
            )

        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized = tokenized.rename_column('label', 'labels')

        train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(tokenized['validation'], batch_size=batch_size)

        # Optimizer and scheduler
        optimizer = AdamW(classifier.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Training loop
        best_val_f1 = 0
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch + 1}/{num_epochs}')

            train_loss, train_acc, train_f1 = train_epoch(
                classifier, train_loader, optimizer, scheduler, classifier.device
            )
            val_loss, val_acc, val_f1 = evaluate(classifier, val_loader, classifier.device)

            logger.info(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}')

            mlflow.log_metrics({
                'train_loss': train_loss, 'train_accuracy': train_acc, 'train_f1': train_f1,
                'val_loss': val_loss, 'val_accuracy': val_acc, 'val_f1': val_f1,
            }, step=epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                classifier.save(output_dir)
                logger.info(f'New best model saved! Val F1: {val_f1:.4f}')

        mlflow.log_metric('best_val_f1', best_val_f1)
        logger.info(f'Training complete. Best Val F1: {best_val_f1:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--dataset_name', default='imdb')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_dir', default='./outputs/model')
    args = parser.parse_args()
    train(**vars(args))
