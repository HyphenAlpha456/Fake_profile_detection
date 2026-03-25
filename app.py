import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
import os
import numpy as np


app = Flask(__name__)


MODEL_FILE = 'model_bundle.joblib'
model = None

try:
    print(f"Loading pre-trained model from '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"FATAL ERROR: '{MODEL_FILE}' not found. Please place your trained model in the project folder.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    """Receives input data, makes a prediction, and returns the result."""
    if model is None:
        return jsonify({'error': 'Model not available. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)
       # input_df = pd.DataFrame([data])
        
        
        for col in ['is_weekend', 'has_media', 'contains_url']:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(bool)

        prediction_result = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        is_fake = bool(prediction_result[1])
        output_class = 'Fake' if is_fake else 'Real'
        confidence_score = np.max(prediction_proba) * 100

        return jsonify({
            'prediction': output_class,
            'confidence': f'{confidence_score:.3f}'
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    if model is not None:
        
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Flask app cannot start because the model failed to load.")
