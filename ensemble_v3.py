import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sentence_transformers import SentenceTransformer
from autoencoder import AutoEncoder
from reader import get_sentence_model
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, ConfusionMatrixDisplay

# Constants
MODELS_DIR = os.path.join("results", "models")
EVAL_DIR = os.path.join("results", "test")
RESULTS_DIR = os.path.join("results", "ensemble_evaluation_v3")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hardcoded model information - simplified approach with correct thresholds from metrics files
# Removed the two underperforming models
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
    "all_MiniLM_L6_v1_AE384": {
        "file_path": os.path.join(MODELS_DIR, "model_all_MiniLM_L6_v1_AE384.pth"),
        "embedding_dim": 384,
        "threshold": 0.0019209596794098616,  # Corrected from metrics.csv
        "transformer_name": "all-MiniLM-L6-v1"
    }
}

# Load test dataset with labels
X_test_df = pd.read_csv("data/final_test.csv")
X_test = X_test_df['prompt'].tolist()
y_test = X_test_df['label'].tolist()

print(f"Loaded test dataset with {len(X_test)} prompts")
print(f"- Malicious prompts: {sum(y_test)}")
print(f"- Benign prompts: {len(y_test) - sum(y_test)}")

# Display some random examples from the dataset
def display_example_prompts(prompts, labels, n=5):
    """Display n random example prompts from the dataset"""
    # Create pairs of prompts and labels
    pairs = list(zip(prompts, labels))
    sample_pairs = random.sample(pairs, min(n, len(pairs)))
    
    print("\n" + "="*80)
    print(f"EXAMPLE PROMPTS ({n} random samples from the dataset):")
    print("="*80)
    
    for i, (prompt, label) in enumerate(sample_pairs):
        # Truncate very long prompts for display
        display_prompt = prompt if len(prompt) < 500 else prompt[:497] + "..."
        label_text = "MALICIOUS" if label == 1 else "BENIGN"
        print(f"\nExample {i+1} [{label_text}]:")
        print(f"{display_prompt}")
    
    print("="*80 + "\n")

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

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }

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

def confidence_weighted_vote(errors, thresholds):
    """Weighted voting based on model confidence (distance from threshold)
    
    For malicious predictions (error < threshold): confidence = threshold/error
    For benign predictions (error >= threshold): confidence = error/threshold
    
    Higher confidence values get more weight in the ensemble decision
    """
    confidence_weights = []
    classifications = []
    
    for error, threshold in zip(errors, thresholds):
        # Determine classification
        classification = 1 if error < threshold else 0
        classifications.append(classification)
        
        # Calculate confidence weight
        if classification == 1:  # malicious prediction
            # Higher ratio means more confidence (error is much smaller than threshold)
            confidence = threshold / error
        else:  # benign prediction
            # Higher ratio means more confidence (error is much larger than threshold)
            confidence = error / threshold
            
        confidence_weights.append(confidence)
    
    # Weighted sum using confidence weights
    weighted_sum = sum(c * w for c, w in zip(classifications, confidence_weights))
    
    # Compare weighted sum to half of total confidence weights
    return 1 if weighted_sum >= sum(confidence_weights) / 2 else 0

# Show how individual models classify example prompts
def display_model_classifications(prompts, true_labels, all_classifications, all_errors, thresholds, model_names, n=2):
    """Display how each individual model classifies n example prompts"""
    # Get indices of correctly and incorrectly classified examples
    correct_indices = []
    incorrect_indices = []
    
    # For each prompt, check if majority vote matches the true label
    for i, (true_label, classifications) in enumerate(zip(true_labels, zip(*all_classifications))):
        majority = majority_vote(classifications)
        if majority == true_label:
            correct_indices.append(i)
        else:
            incorrect_indices.append(i)
    
    # Prioritize showing incorrect classifications
    if incorrect_indices and len(incorrect_indices) >= n//2:
        sample_incorrect = random.sample(incorrect_indices, min(n//2, len(incorrect_indices)))
    else:
        sample_incorrect = incorrect_indices
        
    # Fill the rest with correct classifications
    remaining = n - len(sample_incorrect)
    sample_correct = random.sample(correct_indices, min(remaining, len(correct_indices)))
    
    # Combine samples
    sample_indices = sample_incorrect + sample_correct
    
    print("\n" + "="*80)
    print(f"INDIVIDUAL MODEL CLASSIFICATIONS ({len(sample_indices)} samples):")
    print("="*80)
    
    for idx in sample_indices:
        prompt = prompts[idx]
        true_label = true_labels[idx]
        true_label_text = "MALICIOUS" if true_label == 1 else "BENIGN"
        
        # Truncate very long prompts
        display_prompt = prompt if len(prompt) < 300 else prompt[:297] + "..."
        print(f"\nPrompt: {display_prompt}")
        print(f"True label: {true_label_text}")
        
        # Show classification from each model
        print("\nIndividual Model Classifications:")
        
        # Create a formatted table header
        print(f"{'Model':30} | {'Error':10} | {'Threshold':10} | {'Classification':15} | {'Correct':8}")
        print("-" * 80)
        
        for i, model_name in enumerate(model_names):
            error = all_errors[idx][i]
            threshold = thresholds[i]
            classification = all_classifications[i][idx]
            result = "Malicious" if classification == 1 else "Benign"
            correct = classification == true_label
            correct_mark = "✓" if correct else "✗"
            
            # Format model name to be more readable
            display_name = model_name.replace("_", " ")
            if len(display_name) > 28:
                display_name = display_name[:25] + "..."
                
            print(f"{display_name:30} | {error:.6f} | {threshold:.6f} | {result:15} | {correct_mark:8}")
        
        print("="*80)
    
    print("="*80 + "\n")

# Save ensemble results
def save_ensemble_results(ensemble_name, metrics, y_true, y_pred, all_errors=None):
    """Save ensemble results"""
    results_dir = os.path.join(RESULTS_DIR, ensemble_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    })
    metrics_df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)

    # Create a ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
    
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    
    # Customize the plot (optional)
    plt.title(f'Confusion Matrix - {ensemble_name}')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    
    print(f"Results saved to {results_dir}")

# Save individual model results to a single CSV file
def save_individual_model_results(individual_results, model_names):
    """Save all individual model results to a single CSV file for easy comparison"""
    # Create a DataFrame to store all model results
    data = {
        'model': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'specificity': [],
        'true_positives': [],
        'false_positives': [],
        'true_negatives': [],
        'false_negatives': []
    }
    
    # Create shortened names only for display in plots
    display_names = []
    for name in model_names:
        # For plot display only, shorten if too long
        if len(name) > 20:
            if "MiniLM" in name:
                short_name = f"MiniLM-{name.split('MiniLM')[1][:10]}..."
            elif "mpnet" in name:
                if "multilingual" in name:
                    short_name = "multi-mpnet..."
                elif "multi-qa" in name:
                    short_name = "qa-mpnet..."
                else:
                    short_name = "mpnet..."
            else:
                short_name = name[:17] + "..."
        else:
            short_name = name
        display_names.append(short_name)
    
    # Populate the DataFrame with data from each model
    for i, model_name in enumerate(model_names):
        metrics = individual_results[model_name]
        data['model'].append(model_name)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 
                       'true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
            data[metric].append(metrics[metric])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = os.path.join(RESULTS_DIR, "individual_model_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Individual model results saved to {output_path}")
    
    # Create bar chart visualization for key metrics
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    
    # Create a bar chart comparing models
    plt.figure(figsize=(14, 10))
    
    # Number of models and metrics
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)
    
    # Set up the plot
    width = 0.15  # Width of bars
    x = np.arange(n_models)  # Model positions
    
    # Plot each metric as a group of bars
    for i, metric in enumerate(metrics_to_plot):
        values = []
        for model_name in model_names:
            values.append(individual_results[model_name][metric])
        plt.bar(x + i*width, values, width, label=metric.replace("_", " ").title())
    
    # Add labels and legend
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    # Use shortened names only for plot labels
    plt.xticks(x + width * (n_metrics-1)/2, display_names, rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=n_metrics)
    plt.ylim(0, 1.0)  # Metrics are between 0 and 1
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(RESULTS_DIR, "individual_model_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Individual model comparison chart saved to {plot_path}")

# Save per-prompt model predictions
def save_prompt_predictions(X_test, y_test, model_classifications, model_names, ensemble_methods):
    """Save per-prompt predictions from all models and ensemble methods"""
    # Create a DataFrame with prompts and true labels
    data = {
        'prompt': X_test,
        'true_label': y_test
    }
    
    # Add individual model predictions
    for i, model_name in enumerate(model_names):
        # Get a short version of the model name
        short_name = model_name.replace("all-", "").replace("paraphrase-", "")
        if len(short_name) > 15:
            short_name = short_name[:12] + "..."
        data[f'model_{short_name}'] = model_classifications[i]
    
    # Add ensemble method predictions
    for method_name, predictions in ensemble_methods.items():
        data[f'ensemble_{method_name}'] = predictions
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = os.path.join(RESULTS_DIR, "prompt_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"Per-prompt predictions saved to {output_path}")

def save_ensemble_combined_metrics(ensemble_metrics):
    """Save results from all ensemble methods to a single CSV file for easy comparison"""
    # Create a DataFrame to store all ensemble results
    data = {
        'ensemble_method': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'specificity': [],
        'true_positives': [],
        'false_positives': [],
        'true_negatives': [],
        'false_negatives': []
    }
    
    # Populate the DataFrame with data from each ensemble method
    for method_name, metrics in ensemble_metrics.items():
        data['ensemble_method'].append(method_name)
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 
                       'true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
            data[metric].append(metrics[metric])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = os.path.join(RESULTS_DIR, "ensemble_combined.csv")
    df.to_csv(output_path, index=False)
    print(f"Combined ensemble results saved to {output_path}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Display some example prompts
    display_example_prompts(X_test, y_test, n=6)
    
    # Add CUDA diagnostics
    print("\nPyTorch Info:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print("\nCUDA Diagnostics:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get model names from the hardcoded dictionary
    model_names = get_model_names()
    print(f"Found {len(model_names)} models: {model_names}")
    
    # Load models, thresholds, and sentence transformers
    models = []
    thresholds = []
    sentence_transformers = []
    loaded_model_names = []
    
    for model_name in model_names:
        try:
            print(f"\nLoading model: {model_name}")
            
            # Get transformer name
            transformer_name = get_transformer_name(model_name)
            print(f"  Transformer name: {transformer_name}")
            
            # Get threshold
            threshold = get_threshold(model_name)
            print(f"  Threshold: {threshold:.6f}")
            
            # Load model
            model = load_model(model_name, device)
            
            # Load sentence transformer
            sentence_model = get_sentence_model(device, model_name=transformer_name)
            
            # Store everything if successful
            models.append(model)
            thresholds.append(threshold)
            sentence_transformers.append(sentence_model)
            loaded_model_names.append(model_name)
            
        except Exception as e:
            print(f"  Error loading model {model_name}: {e}")
            print(f"  Skipping this model and continuing with others...")
            continue
    
    if len(models) == 0:
        print("No models could be loaded successfully. Exiting.")
        return
        
    print(f"\nSuccessfully loaded {len(models)} models.")
    
    # Calculate weights for weighted voting (inverse of threshold)
    # Lower threshold = better performance on malicious samples
    weights = [1/t for t in thresholds]
    
    # Process test data
    all_errors = []
    model_classifications = [[] for _ in range(len(models))]
    
    print(f"\nProcessing {len(X_test)} test prompts...")
    
    for i, prompt in enumerate(X_test):
        prompt_errors = []
        
        for j, (model, sentence_model, threshold) in enumerate(zip(models, sentence_transformers, thresholds)):
            error, classification = classify_prompt(prompt, model, sentence_model, threshold, device)
            prompt_errors.append(error)
            model_classifications[j].append(classification)
        
        all_errors.append(prompt_errors)
    
    # Display examples of individual model classifications
    display_model_classifications(X_test, y_test, model_classifications, all_errors, thresholds, loaded_model_names, n=4)
    
    # Apply ensemble methods
    majority_classifications = []
    weighted_classifications = []
    avg_threshold_classifications = []
    min_error_classifications = []
    confidence_weighted_classifications = []
    
    for i, prompt in enumerate(X_test):
        # Get errors and classifications for this prompt across all models
        errors = all_errors[i]
        classifications = [model_cls[i] for model_cls in model_classifications]
        
        # Apply different ensemble methods
        majority_classifications.append(majority_vote(classifications))
        weighted_classifications.append(weighted_vote(classifications, weights))
        avg_threshold_classifications.append(threshold_average(errors, thresholds))
        min_error_classifications.append(min_error_ensemble(errors, thresholds))
        confidence_weighted_classifications.append(confidence_weighted_vote(errors, thresholds))
    
    # Calculate metrics for each ensemble method
    ensemble_methods = {
        "majority_vote": majority_classifications,
        "weighted_vote": weighted_classifications,
        "avg_threshold": avg_threshold_classifications,
        "min_error": min_error_classifications,
        "confidence_weighted": confidence_weighted_classifications
    }
    
    # Also evaluate individual models
    print("\nIndividual Model Results:")
    print(f"{'Model':30} | {'Accuracy':8} | {'Precision':9} | {'Recall':8} | {'F1 Score':8} | {'Specificity':11}")
    print("-" * 80)
    
    individual_results = {}
    
    for i, model_name in enumerate(loaded_model_names):
        metrics = calculate_metrics(y_test, model_classifications[i])
        individual_results[model_name] = metrics
        
        # Format model name to be more readable
        display_name = model_name.replace("_", " ")
        if len(display_name) > 28:
            display_name = display_name[:25] + "..."
            
        print(f"{display_name:30} | {metrics['accuracy']:.6f} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1_score']:.6f} | {metrics['specificity']:.6f}")
    
    # Save individual model results to a single CSV file
    save_individual_model_results(individual_results, loaded_model_names)
    
    # Save per-prompt predictions for detailed analysis
    save_prompt_predictions(X_test, y_test, model_classifications, loaded_model_names, ensemble_methods)
    
    print("\nEnsemble Method Results:")
    ensemble_metrics = {}
    for method_name, classifications in ensemble_methods.items():
        # Calculate comprehensive metrics
        metrics = calculate_metrics(y_test, classifications)
        ensemble_metrics[method_name] = metrics
        
        print(f"\n{method_name} Ensemble:")
        print(f"Accuracy: {metrics['accuracy']:.6f}")
        print(f"Precision: {metrics['precision']:.6f}")
        print(f"Recall: {metrics['recall']:.6f}")
        print(f"F1 Score: {metrics['f1_score']:.6f}")
        print(f"Specificity: {metrics['specificity']:.6f}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"True Negatives: {metrics['true_negatives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        
        # Save detailed results
        save_ensemble_results(method_name, metrics, y_test, classifications, all_errors if method_name == "avg_threshold" else None)
    
    # Save combined ensemble metrics
    save_ensemble_combined_metrics(ensemble_metrics)
    
    # Create comparison bar chart for ensemble methods
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Number of metrics to plot
    num_metrics = len(metrics_to_plot)
    num_methods = len(ensemble_methods)
    
    # Set width of bars
    bar_width = 0.15  # Reduced from 0.2 to fit the 5th method
    index = np.arange(num_metrics)
    
    # Plot bars for each ensemble method
    for i, (method, metrics) in enumerate(ensemble_metrics.items()):
        values = [metrics[m] for m in metrics_to_plot]
        ax.bar(index + i*bar_width, values, bar_width, label=method.replace('_', ' ').title())
    
    # Add labels and title
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Ensemble Methods')
    ax.set_xticks(index + bar_width * (num_methods-1)/2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ensemble_comparison.png'))
    print(f"\nComparison plot saved to {os.path.join(RESULTS_DIR, 'ensemble_comparison.png')}")

if __name__ == "__main__":
    main() 