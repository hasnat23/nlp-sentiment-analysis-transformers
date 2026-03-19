import torch
import numpy as np
from typing import List, Union, Dict
import logging
from model import SentimentClassifier

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Inference class for sentiment analysis."""

    def __init__(self, model_dir: str, num_labels: int = 2, device: str = None):
        self.classifier = SentimentClassifier.load(
            model_dir=model_dir,
            num_labels=num_labels,
            device=device,
        )
        self.num_labels = num_labels
        self.label_map = {0: 'NEGATIVE', 1: 'POSITIVE'} if num_labels == 2 else {
            i: f'LABEL_{i}' for i in range(num_labels)
        }

    def predict(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[Dict]:
        """Predict sentiment for one or more texts."""
        if isinstance(texts, str):
            texts = [texts]

        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.classifier.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )

            with torch.no_grad():
                outputs = self.classifier.model(
                    input_ids=encodings['input_ids'].to(self.classifier.device),
                    attention_mask=encodings['attention_mask'].to(self.classifier.device),
                )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            for text, pred, prob in zip(batch_texts, preds, probs):
                all_results.append({
                    'text': text,
                    'label': self.label_map[int(pred)],
                    'label_id': int(pred),
                    'confidence': float(prob[pred]),
                    'probabilities': {self.label_map[i]: float(prob[i]) for i in range(self.num_labels)},
                })

        return all_results

    def predict_single(self, text: str) -> Dict:
        """Predict sentiment for a single text string."""
        results = self.predict([text])
        return results[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='Path to saved model')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--num_labels', type=int, default=2)
    args = parser.parse_args()

    predictor = SentimentPredictor(model_dir=args.model_dir, num_labels=args.num_labels)

    if args.text:
        result = predictor.predict_single(args.text)
        print(f"Text: {result['text']}")
        print(f"Label: {result['label']} (confidence: {result['confidence']:.4f})")
        print(f"Probabilities: {result['probabilities']}")
    else:
        # Demo mode
        demo_texts = [
            'This movie was absolutely fantastic! I loved every moment.',
            'The product quality is terrible. Complete waste of money.',
            'Average experience, nothing special but not bad either.',
        ]
        results = predictor.predict(demo_texts)
        for r in results:
            print(f"[{r['label']}] ({r['confidence']:.2%}) - {r['text'][:60]}")
