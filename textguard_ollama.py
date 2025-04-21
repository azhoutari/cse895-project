import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from autoencoder import AutoEncoder
import requests
import json

# Constants
MODELS_DIR = os.path.join("results", "models")
RESULTS_DIR = os.path.join("results", "ensemble_evaluation_v2")

# Hardcoded model information - simplified approach with correct thresholds from metrics files
MODELS = {
    "all_MiniLM_L6_v2_AE384": {
        "file_path": os.path.join(MODELS_DIR, "model_all_MiniLM_L6_v2_AE384.pth"),
        "embedding_dim": 384,
        "threshold": 0.001875142683275044,  # Verified from metrics.csv
        "transformer_name": "all-MiniLM-L6-v2"
    },
    "all_mpnet_base_v2_AE768": {
        "file_path": os.path.join(MODELS_DIR, "model_all_mpnet_base_v2_AE768.pth"),
        "embedding_dim": 768,
        "threshold": 0.0009223620872944593,  # Corrected from metrics.csv
        "transformer_name": "all-mpnet-base-v2"
    },
    "paraphrase_multilingual_MiniLM_L12_v2_AE384": {
        "file_path": os.path.join(MODELS_DIR, "model_paraphrase_multilingual_MiniLM_L12_v2_AE384.pth"),
        "embedding_dim": 384,
        "threshold": 0.030461249873042107,  # Corrected from metrics.csv
        "transformer_name": "paraphrase-multilingual-MiniLM-L12-v2"
    },
    "all_distilroberta_v1_AE768": {
        "file_path": os.path.join(MODELS_DIR, "model_all_distilroberta_v1_AE768.pth"),
        "embedding_dim": 768,
        "threshold": 0.0009499453008174896,  # Corrected from metrics.csv
        "transformer_name": "all-distilroberta-v1"
    },
    "all_MiniLM_L12_v2_AE384": {
        "file_path": os.path.join(MODELS_DIR, "model_all_MiniLM_L12_v2_AE384.pth"),
        "embedding_dim": 384, 
        "threshold": 0.0018401979468762875,  # Corrected from metrics.csv
        "transformer_name": "all-MiniLM-L12-v2"
    },
    "paraphrase_albert_small_v2_AE768": {
        "file_path": os.path.join(MODELS_DIR, "model_paraphrase_albert_small_v2_AE768.pth"),
        "embedding_dim": 768,
        "threshold": 0.14304590225219727,  # Corrected from metrics.csv
        "transformer_name": "paraphrase-albert-small-v2"
    },
    "paraphrase_multilingual_mpnet_base_v2_AE768": {
        "file_path": os.path.join(MODELS_DIR, "model_paraphrase_multilingual_mpnet_base_v2_AE768.pth"),
        "embedding_dim": 768,
        "threshold": 0.006693185772746801,  # Corrected from metrics.csv
        "transformer_name": "paraphrase-multilingual-mpnet-base-v2"
    },
    "all_roberta_large_v1_AE1024": {
        "file_path": os.path.join(MODELS_DIR, "model_all_roberta_large_v1_AE1024.pth"),
        "embedding_dim": 1024,
        "threshold": 0.0007011170382611454,  # Corrected from metrics.csv
        "transformer_name": "all-roberta-large-v1"
    },
    "multi_qa_mpnet_base_dot_v1_AE768": {
        "file_path": os.path.join(MODELS_DIR, "model_multi_qa_mpnet_base_dot_v1_AE768.pth"),
        "embedding_dim": 768,
        "threshold": 0.02795509621500969,  # Corrected from metrics.csv
        "transformer_name": "multi-qa-mpnet-base-dot-v1"
    },
    "all_MiniLM_L6_v1_AE384": {
        "file_path": os.path.join(MODELS_DIR, "model_all_MiniLM_L6_v1_AE384.pth"),
        "embedding_dim": 384,
        "threshold": 0.0019209596794098616,  # Corrected from metrics.csv
        "transformer_name": "all-MiniLM-L6-v1"
    }
}

# CUDA setup
print("\nSetting up CUDA...")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simplified model retrieval functions
def get_model_names():
    """Return list of model names"""
    return list(MODELS.keys())

def load_model(model_name, device):
    """Load model directly using hardcoded information"""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in hardcoded models")
    
    model_info = MODELS[model_name]
    model = AutoEncoder(input_dim=model_info["embedding_dim"]).to(device)
    
    try:
        model.load_state_dict(torch.load(model_info["file_path"], map_location=device))
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {model_info['file_path']}: {e}")

def get_threshold(model_name):
    """Return hardcoded threshold"""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in hardcoded models")
    return MODELS[model_name]["threshold"]

def get_transformer_name(model_name):
    """Return the actual transformer name"""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in hardcoded models")
    return MODELS[model_name]["transformer_name"]

# Get reconstruction error for a prompt using a specific model
def get_reconstruction_error(prompt, model, sentence_model, device):
    emb = sentence_model.encode(prompt, convert_to_tensor=True).clone().detach().to(device)
    emb = emb.unsqueeze(0)
    
    with torch.no_grad():
        recon = model(emb)
        error = torch.mean((recon - emb) ** 2, dim=1).item()
    
    return error

# Classify prompt with a specific model
def classify_prompt(prompt, model, sentence_model, threshold, device):
    error = get_reconstruction_error(prompt, model, sentence_model, device)
    classification = 1 if error < threshold else 0
    return error, classification

# Ensemble methods
def majority_vote(classifications):
    """Simple majority voting"""
    votes = sum(classifications)
    return 1 if votes >= len(classifications) / 2 else 0

def weighted_vote(classifications, weights):
    """Weighted voting based on model performance"""
    weighted_sum = sum(c * w for c, w in zip(classifications, weights))
    return 1 if weighted_sum >= sum(weights) / 2 else 0

def threshold_average(errors, thresholds):
    """Average error compared to average threshold"""
    avg_error = sum(errors) / len(errors)
    avg_threshold = sum(thresholds) / len(thresholds)
    return 1 if avg_error < avg_threshold else 0

def min_error_ensemble(errors, thresholds):
    """Classify as malicious if any model's error is below its threshold"""
    for error, threshold in zip(errors, thresholds):
        if error < threshold:
            return 1
    return 0

# Function to connect to Ollama API
def query_ollama(prompt, model="granite3.3:2b", system_prompt=None):
    """Send a prompt to Ollama API and get the response"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    # Add system prompt if provided
    if system_prompt:
        data["system"] = system_prompt
    
    try:
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return "Error connecting to Ollama"
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return "Error connecting to Ollama"

def main():
    # Get model names from the hardcoded dictionary
    model_names = get_model_names()
    print(f"Found {len(model_names)} models")
    
    print("\nLoading models... (this may take a moment)")
    
    # Load models, thresholds, and sentence transformers
    models = []
    thresholds = []
    sentence_transformers = []
    loaded_model_names = []
    
    for model_name in model_names:
        try:
            # Get transformer name
            transformer_name = get_transformer_name(model_name)
            
            # Get threshold
            threshold = get_threshold(model_name)
            
            # Load model
            model = load_model(model_name, device)
            
            # Load sentence transformer
            sentence_model = SentenceTransformer(transformer_name)
            sentence_model = sentence_model.to(device)
            
            # Store everything if successful
            models.append(model)
            thresholds.append(threshold)
            sentence_transformers.append(sentence_model)
            loaded_model_names.append(model_name)
            
            print(f"  Loaded model: {model_name}")
            
        except Exception as e:
            print(f"  Error loading model {model_name}: {e}")
            print(f"  Skipping this model and continuing with others...")
            continue
    
    if len(models) == 0:
        print("No models could be loaded successfully. Exiting.")
        return
        
    print(f"Successfully loaded {len(models)} models.\n")
    
    # Calculate weights for weighted voting (inverse of threshold)
    # Lower threshold = better performance on malicious samples
    weights = [1/t for t in thresholds]
    
    # Set default Ollama model
    ollama_model = "granite3.3:2b"
    
    # Get user input
    print("=" * 80)
    print("TextGuard + Ollama Integration")
    print("This tool will check if prompts are potentially harmful before sending to Ollama")
    print("=" * 80)
    print("Type 'exit' to quit")
    print("Type 'model [name]' to change the Ollama model")
    
    while True:
        # Get prompt from user
        prompt = input("\nYou: ")
        
        if prompt.lower() == 'exit':
            print("Exiting. Goodbye!")
            break
        
        if prompt.lower().startswith('model '):
            new_model = prompt[6:].strip()
            print(f"Changing Ollama model to: {new_model}")
            ollama_model = new_model
            continue
        
        if not prompt.strip():
            print("Please enter a valid prompt.")
            continue
        
        print("\nAnalyzing prompt...")
        
        # Process the prompt with each model
        prompt_errors = []
        model_classifications = []
        
        print("\nIndividual Model Results:")
        print(f"{'Model':30} | {'Error':10} | {'Threshold':10} | {'Classification':15}")
        print("-" * 80)
        
        for i, (model, sentence_model, threshold, model_name) in enumerate(zip(models, sentence_transformers, thresholds, loaded_model_names)):
            error, classification = classify_prompt(prompt, model, sentence_model, threshold, device)
            prompt_errors.append(error)
            model_classifications.append(classification)
            
            # Format model name to be more readable
            display_name = model_name.replace("_", " ")
            if len(display_name) > 28:
                display_name = display_name[:25] + "..."
                
            result = "Malicious" if classification == 1 else "Benign"
            print(f"{display_name:30} | {error:.6f} | {threshold:.6f} | {result:15}")
        
        # Apply weighted vote ensemble method (as requested)
        weighted_result = weighted_vote(model_classifications, weights)
        
        # Count individual model votes
        malicious_votes = sum(model_classifications)
        benign_votes = len(model_classifications) - malicious_votes
        
        # Display ensemble results
        print("\nEnsemble Method Results:")
        print("-" * 80)
        print(f"Models voting Malicious: {malicious_votes}/{len(model_classifications)}")
        print(f"Models voting Benign: {benign_votes}/{len(model_classifications)}")
        print("-" * 80)
        print(f"Weighted Vote: {'Malicious' if weighted_result == 1 else 'Benign'}")
        
        # Define the system prompt for Ollama
        system_prompt = "You are a helpful AI assistant. Answer the user's questions accurately and concisely."
        
        # Check if prompt should be blocked (weighted vote == 1 means malicious)
        if weighted_result == 1:
            print("\nBlocked by TextGuard")
            print("This prompt was identified as potentially harmful.")
        else:
            # If the prompt is safe, send it to Ollama
            print("\nPrompt passed TextGuard safety check. Sending to Ollama...")
            response = query_ollama(prompt, model=ollama_model, system_prompt=system_prompt)
            print(f"\nOllama: {response}")

if __name__ == "__main__":
    main() 