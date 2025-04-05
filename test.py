import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from reader import to_embeddings, get_loader, get_sentence_model
from autoencoder import AutoEncoder
import os
from datasets import load_dataset

ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")

def classify_prompt(prompt, model, sentence_model, threshold, device):
    emb = sentence_model.encode(prompt, convert_to_tensor=True).clone().detach().to(device)
    emb = emb.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        recon = model(emb)
        error = torch.mean((recon - emb) ** 2, dim=1).item()

    classification = 1 if error < threshold else 0

    return error, classification

def save_results(model_name, reconstruction_errors, recall, fnr, threshold, X_test, classifications, errors, directory):
    # Create results directory with model name
    results_dir = f'{directory}/{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'metric': ['recall', 'false_negative_rate', 'threshold', 'mean_error', 'std_error'],
        'value': [recall, fnr, threshold, np.mean(reconstruction_errors), np.std(reconstruction_errors)]
    })
    metrics_df.to_csv(f'{results_dir}/metrics.csv', index=False)
    
    # Save detailed results including prompts, errors, and classifications
    results_df = pd.DataFrame({
        'prompt': X_test,
        'reconstruction_error': errors,
        'classification': classifications,
        'classified_as': ['Malicious' if c == 1 else 'Benign' for c in classifications]
    })
    results_df.to_csv(f'{results_dir}/errors.csv', index=False)
    
    # Save histogram
    plt.figure(figsize=(8, 4))
    plt.hist(reconstruction_errors, bins=30, edgecolor='black')
    plt.title(f"Histogram of Reconstruction Errors on Test Set ({model_name})")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.savefig(f'{results_dir}/histogram.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    X_eval = pd.read_csv("data/test.csv")['prompt'].tolist()
    
    # Set model name (should match the one used in training)
    model_name = "all-MiniLM-L6-v2"
    
    # Generate embeddings
    eval_embeddings = to_embeddings(X_eval, device, model_name=model_name)
    eval_loader = get_loader(eval_embeddings, model_name=model_name)
    
    # Load the trained model
    input_dim = eval_embeddings.shape[1]
    model = AutoEncoder(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load("checkpoint.pth", weights_only=True))
    
    # Calculate reconstruction errors
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            recon_batch = model(batch)
            batch_errors = torch.mean((recon_batch - batch) ** 2, dim=1)
            all_errors.extend(batch_errors.cpu().numpy())
    
    reconstruction_errors = np.array(all_errors)
    print(f"Average Reconstruction Error on Test Set: {np.mean(reconstruction_errors):.4f}")

    threshold_percentile = 95 
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    print("Threshold set at the {}th percentile: {:.4f}".format(threshold_percentile, threshold))
    
    # Get sentence model for classification
    sentence_model = get_sentence_model(device, model_name=model_name)
    
    # Perform classification
    errors = []
    classifications = []
    
    for prompt in X_eval:
        error, classification = classify_prompt(prompt, model, sentence_model, threshold, device)
        errors.append(error)
        classifications.append(classification)
    
    # Calculate metrics
    false_negatives = []
    for prompt, classification, error in zip(X_eval, classifications, errors):
        if classification == 0:
            false_negatives.append((prompt, error))
    
    FN = len(false_negatives)
    TP = len(X_eval) - FN
    
    # Calculate Recall and FNR
    recall = TP / (TP + FN)
    fnr = FN / (TP + FN)
    
    # Save all results
    save_results(model_name, reconstruction_errors, recall, fnr, threshold, X_eval, classifications, errors, f"results/eval")

    errors = []
    classifications = []
    
    X_test = ds['harmful']['Goal']
    
    for prompt in X_test:
        error, classification = classify_prompt(prompt, model, sentence_model, threshold, device)
    
        errors.append(error)
        classifications.append(classification)
    
    false_negatives = []
    for prompt, classification, error in zip(X_test, classifications, errors):
        if classification == 0:
            false_negatives.append((prompt, error))
    
    FN = len(false_negatives)
    TP = len(X_test) - FN
    
    # Calculate Recall and FNR
    recall = TP / (TP + FN)
    fnr = FN / (TP + FN)

    save_results(model_name, reconstruction_errors, recall, fnr, threshold, X_test, classifications, errors, f"results/test")


if __name__ == "__main__":
    main() 