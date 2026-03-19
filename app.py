from flask import Flask, request, jsonify
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import SentimentPredictor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model on startup
MODEL_DIR = os.environ.get('MODEL_DIR', './outputs/model')
NUM_LABELS = int(os.environ.get('NUM_LABELS', 2))

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        logger.info(f'Loading model from {MODEL_DIR}')
        predictor = SentimentPredictor(model_dir=MODEL_DIR, num_labels=NUM_LABELS)
        logger.info('Model loaded successfully')
    return predictor


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_dir': MODEL_DIR})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Request must include a "text" field'}), 400

    text = data['text']
    p = get_predictor()

    if isinstance(text, list):
        results = p.predict(text)
    else:
        results = p.predict([text])

    return jsonify({'results': results})


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Request must include a "texts" field (list)'}), 400

    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({'error': '"texts" must be a list of strings'}), 400

    p = get_predictor()
    results = p.predict(texts, batch_size=data.get('batch_size', 32))
    return jsonify({'results': results, 'count': len(results)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
