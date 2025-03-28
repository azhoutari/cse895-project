import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from reader import to_embeddings, get_loader, get_sentence_model
from autoencoder import AutoEncoder
import os

def classify_prompt(prompt, model, sentence_model, threshold, device):
    emb = sentence_model.encode(prompt, convert_to_tensor=True).clone().detach().to(device)
    emb = emb.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        recon = model(emb)
        error = torch.mean((recon - emb) ** 2, dim=1).item()

    classification = 1 if error < threshold else 0

    return error, classification

def save_results(model_name, reconstruction_errors, recall, fnr, threshold, X_test, classifications, errors):
    # Create results directory with model name
    results_dir = f'test_results_{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'metric': ['recall', 'false_negative_rate', 'threshold', 'mean_error', 'std_error'],
        'value': [recall, fnr, threshold, np.mean(reconstruction_errors), np.std(reconstruction_errors)]
    })
    metrics_df.to_csv(f'{results_dir}/metrics_{model_name}.csv', index=False)
    
    # Save detailed results including prompts, errors, and classifications
    results_df = pd.DataFrame({
        'prompt': X_test,
        'reconstruction_error': errors,
        'classification': classifications,
        'classified_as': ['Malicious' if c == 1 else 'Benign' for c in classifications]
    })
    results_df.to_csv(f'{results_dir}/errors_{model_name}.csv', index=False)
    
    # Save histogram
    plt.figure(figsize=(8, 4))
    plt.hist(reconstruction_errors, bins=30, edgecolor='black')
    plt.title(f"Histogram of Reconstruction Errors on Test Set ({model_name})")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.savefig(f'{results_dir}/histogram_{model_name}.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    X_test = pd.read_csv("data/test.csv")['prompt'].tolist()
    
    # Set model name (should match the one used in training)
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Generate embeddings
    test_embeddings = to_embeddings(X_test, device, model_name=model_name)
    test_loader = get_loader(test_embeddings, model_name=model_name)
    
    # Load the trained model
    input_dim = test_embeddings.shape[1]
    model = AutoEncoder(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load("checkpoint.pth", weights_only=True))
    
    # Calculate reconstruction errors
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            recon_batch = model(batch)
            batch_errors = torch.mean((recon_batch - batch) ** 2, dim=1)
            all_errors.extend(batch_errors.cpu().numpy())
    
    reconstruction_errors = np.array(all_errors)
    print(f"Average Reconstruction Error on Test Set: {np.mean(reconstruction_errors):.4f}")
    
    # Set threshold
    threshold = 0.002  # Fixed threshold as in playground.ipynb
    print(f"Using threshold: {threshold:.4f}")
    
    # Get sentence model for classification
    sentence_model = get_sentence_model(device, model_name=model_name)
    
    # Perform classification
    errors = []
    classifications = []
    
    for prompt in X_test:
        error, classification = classify_prompt(prompt, model, sentence_model, threshold, device)
        errors.append(error)
        classifications.append(classification)
    
    # Calculate metrics
    false_negatives = []
    for prompt, classification, error in zip(X_test, classifications, errors):
        if classification == 0:
            false_negatives.append((prompt, error))
    
    FN = len(false_negatives)
    TP = len(X_test) - FN
    
    # Calculate Recall and FNR
    recall = TP / (TP + FN)
    fnr = FN / (TP + FN)
    
    print(f"Recall: {recall:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    
    # Save all results
    save_results(model_name, reconstruction_errors, recall, fnr, threshold, X_test, classifications, errors)

if __name__ == "__main__":
    main() 