# utils/model_loader.py
import os
import sys
import boto3
import tempfile
import importlib.util

def download_file_from_s3(bucket_name, file_key, local_path=None):
    """Download file from S3 bucket"""
    if local_path is None:
        # Create a temporary file if no path specified
        temp_dir = tempfile.gettempdir()
        local_path = os.path.join(temp_dir, os.path.basename(file_key))
    
    s3_client = boto3.client('s3')
    
    # Download the file if it doesn't exist locally
    if not os.path.exists(local_path):
        print(f"Downloading file from s3://{bucket_name}/{file_key} to {local_path}")
        s3_client.download_file(bucket_name, file_key, local_path)
    else:
        print(f"Using existing file at {local_path}")
    
    return local_path

def load_model(bucket_name, model_script_key, weights_key=None):
    """
    Load model from S3:
    1. Download the model Python script
    2. Import it as a module
    3. If weights_key is provided, download the weights file
    4. Return the model with weights loaded
    """
    # Download the model script
    model_script_path = download_file_from_s3(bucket_name, model_script_key)
    
    # Import the model script as a module
    module_name = os.path.basename(model_script_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, model_script_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = model_module
    spec.loader.exec_module(model_module)
    
    # Get the model using the get_model function
    if hasattr(model_module, 'get_model'):
        model = model_module.get_model()
    else:
        raise AttributeError("The model module does not have a get_model function")
    
    # If weights are provided, download and load them
    if weights_key:
        weights_path = download_file_from_s3(bucket_name, weights_key)
        
        # Load weights - assuming it's a PyTorch model
        import torch
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    return model