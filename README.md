# NLP Sentiment Analysis with Transformers

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive NLP pipeline for **multi-class sentiment analysis** using fine-tuned BERT and RoBERTa models. This project covers end-to-end model training, evaluation, and deployment using Hugging Face Transformers, PyTorch, and MLflow for experiment tracking.

## Project Overview

This project fine-tunes pre-trained transformer models (BERT, RoBERTa) for sentiment classification across three domains:
- **Product Reviews** (Amazon dataset) — Positive / Neutral / Negative
- **Social Media** (Twitter/X) — Sentiment + Emotion detection
- **Financial News** (FinancialPhraseBank) — Bullish / Bearish / Neutral

## Key Features

- Fine-tuned `bert-base-uncased` and `roberta-base` on domain-specific datasets
- Custom training loop with mixed-precision (FP16) using `accelerate`
- Hyperparameter optimization with Optuna
- MLflow experiment tracking and model registry
- REST API deployment with FastAPI
- Model quantization for efficient inference (INT8)
- Evaluation: Accuracy, F1-Score, AUC-ROC, Confusion Matrix

## Tech Stack

| Category | Tools |
|---|---|
| Framework | PyTorch, Hugging Face Transformers |
| Training | Accelerate, PEFT (LoRA), Optuna |
| Tracking | MLflow, Weights & Biases |
| Deployment | FastAPI, Docker |
| Data | Hugging Face Datasets, Pandas |
| Evaluation | Scikit-learn, Matplotlib, Seaborn |

## Model Performance

| Model | Dataset | Accuracy | F1-Score |
|---|---|---|---|
| BERT-base | Amazon Reviews | 92.4% | 0.921 |
| RoBERTa-base | FinancialPhraseBank | 94.1% | 0.938 |
| RoBERTa-base | Twitter Sentiment | 88.7% | 0.884 |

## Project Structure

```
nlp-sentiment-analysis-transformers/
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Tokenized & preprocessed
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Training.ipynb     # Model fine-tuning
│   └── 03_Evaluation.ipynb   # Results & visualizations
├── src/
│   ├── data_processing.py    # Dataset loading & tokenization
│   ├── model.py              # Model architecture & config
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Evaluation metrics
│   └── inference.py          # Inference pipeline
├── api/
│   └── main.py               # FastAPI inference endpoint
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/hasnat23/nlp-sentiment-analysis-transformers.git
cd nlp-sentiment-analysis-transformers

# Install dependencies
pip install -r requirements.txt

# Fine-tune BERT on Amazon Reviews
python src/train.py --model bert-base-uncased --dataset amazon_polarity --epochs 3 --lr 2e-5

# Run inference
python src/inference.py --text "This product is absolutely amazing!"
```

## Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)
```

## Results & Visualizations

Detailed evaluation results, confusion matrices, and training curves are available in the `notebooks/03_Evaluation.ipynb` notebook.

## Author

**Muhammad Hasnat**  
ML & AI Engineer | NLP Specialist  
[LinkedIn](https://linkedin.com/in/hasnat23) | [GitHub](https://github.com/hasnat23)

## License

MIT License — see [LICENSE](LICENSE) for details.
