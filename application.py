# application.py
import os
from utils.image_processing import process_image, scale_bbox
from flask import Flask, request, jsonify
from utils.model_loader import load_model
from flask_cors import CORS
import torch
import logging

# Initialize Flask application
application = Flask(__name__)
application.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024 
CORS(application, origins=["https://bzgarden.org", "https://facedetection.bzgarden.org", "https://www.facedetection.bzgarden.org"])
application.logger.setLevel(logging.INFO)

# Define S3 information with defaults
S3_BUCKET = os.environ.get('MODEL_S3_BUCKET', 'face-detection-dataset')
MODEL_SCRIPT_KEY = os.environ.get('MODEL_SCRIPT_KEY', 'FaceData/model/model.py')
MODEL_WEIGHTS_KEY = os.environ.get('MODEL_WEIGHTS_KEY', 'FaceData/output/model_20250413_003714.pth')

# Global variable for the model
model = None

@application.before_first_request
def load_ml_model():
    """Load the model before the first request"""
    global model
    try:
        # Load model with weights
        model = load_model(S3_BUCKET, MODEL_SCRIPT_KEY, MODEL_WEIGHTS_KEY)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@application.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.data

        if not data:
            return jsonify({'error': 'No data provided'}), 400
        processed_image, org_X, org_Y = process_image(data)
        application.logger.info(org_X)
        application.logger.info(org_Y)
        application.logger.info(processed_image.shape)

        try:
            model.eval()
            prediction = model(processed_image.unsqueeze(0))
            
        except Exception as model_e:
            application.logger.error(f"Error during model prediction: {model_e}")
            return jsonify({'error': f"Model prediction failed: {str(model_e)}"}, 500)

        try:
            serialized_prediction = []
            for item in prediction:
                serialized_item = {}
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        serialized_item[key] = value.tolist()
                    else:
                        serialized_item[key] = value
                serialized_prediction.append(serialized_item)

            scaled_bboxes = []
            for bbox in serialized_prediction[0]['boxes']:
                scaled_bboxes.append(scale_bbox(bbox, org_X, org_Y))
            serialized_prediction[0]['boxes'] = scaled_bboxes

            result = {
                'prediction': serialized_prediction,
                'status': 'success'
            }
        except Exception as e:
            application.logger.error(e)

        application.logger.info(f"Prediction output: {serialized_prediction}")
        application.logger.info('checkpoint E')

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@application.route('/')
def home():
    return "Face Detection Service"
