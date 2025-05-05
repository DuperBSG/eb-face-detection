# 👤 EB Face Detection

A Flask-based face detection service deployed on AWS Elastic Beanstalk. This application uses a custom PyTorch model to detect faces in images.

## 📋 Overview

This service provides a REST API endpoint for face detection in images. The application loads a pre-trained face detection model from AWS S3 and exposes an endpoint that accepts image data, processes it, and returns face detection results.

## ✨ Features

- 🔍 Face detection using a custom PyTorch model
- 🌐 REST API endpoint for image processing
- ☁️ AWS Elastic Beanstalk deployment configuration
- 🔒 CORS support for specified domains
- 🖼️ Image preprocessing and result scaling

## 🛠️ Tech Stack

- 🐍 Python 3
- 🌶️ Flask web framework
- 🔥 PyTorch for deep learning
- 📷 OpenCV for image processing
- 🚀 AWS Elastic Beanstalk for deployment
- 🗄️ AWS S3 for model storage

## 🔌 API Endpoints

### Predict Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Body**: Binary image data
- **Response**: JSON containing detection results, including bounding boxes and confidence scores

### Home Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Response**: Simple text message indicating the service is running

## 🚀 Deployment

This application is configured for deployment on AWS Elastic Beanstalk. The necessary configuration files are provided in the `.ebextensions` and `.platform` directories.

### 🔧 Environment Variables

The following environment variables can be configured:

- `MODEL_S3_BUCKET`: S3 bucket containing the model (default: 'face-detection-dataset')
- `MODEL_SCRIPT_KEY`: Path to the model script in S3 (default: 'FaceData/model/model.py')
- `MODEL_WEIGHTS_KEY`: Path to the model weights in S3 (default: 'FaceData/output/model_20250413_003714.pth')

## 💻 Local Development

1. Create a virtual environment:

   ```
   python -m venv virt
   source virt/bin/activate
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```
   python application.py
   ```

## 📁 Project Structure

- `application.py`: Main Flask application
- `utils/`: Utility modules
  - `image_processing.py`: Image preprocessing functions
  - `model_loader.py`: Functions to load the model from S3
  - `visualize.py`: Visualization utilities
- `.ebextensions/`: Elastic Beanstalk configuration
- `.platform/`: Platform configuration
- `requirements.txt`: Python dependencies
