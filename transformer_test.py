import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformer_autoencoder import TransformerAutoEncoder

# Function to compute the reconstruction loss for one prompt
def classify_prompt(prompt, transformer_ae, threshold, device):
    transformer_ae.model.eval()
    with torch.no_grad():
        inputs = transformer_ae.tokenize_prompts([prompt])
        outputs = transformer_ae.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    classification = 1 if loss < threshold else 0
    return loss, classification

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/bart-base"
    max_length = 128

    # Load test data (assumes a CSV with a 'prompt' column)
    X_test = pd.read_csv("data/test.csv")['prompt'].tolist()
    
    # Initialize the transformer autoencoder and load trained weights
    transformer_ae = TransformerAutoEncoder(model_name=model_name, device=device, max_length=max_length)
    transformer_ae.model.load_state_dict(torch.load("transformer_checkpoint.pth", map_location=device))
    
    # Set a threshold (this value will need tuning based on your validation data)
    threshold = 2.0
    
    errors = []
    classifications = []
    
    for prompt in X_test:
        loss, classification = classify_prompt(prompt, transformer_ae, threshold, device)
        errors.append(loss)
        classifications.append(classification)
    
    reconstruction_errors = np.array(errors)
    print(f"Average Reconstruction Loss on Test Set: {np.mean(reconstruction_errors):.4f}")
    
    # (You can add further steps to save results or plot a histogram similar to your test.py)
