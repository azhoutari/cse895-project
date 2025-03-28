import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

def to_embeddings(data, device, model_name="all-MiniLM-L6-v2"):
    """
    Convert a list of text data into embeddings using a sentence transformer model.
    
    Args:
        data (list): List of text data to convert into embeddings
        device (torch.device): Device to run the model on
        model_name (str, optional): Name of the sentence transformer model to use. 
                                  Defaults to "all-MiniLM-L6-v2"
    
    Returns:
        torch.Tensor: Tensor containing the embeddings
    """
    sentence_model = SentenceTransformer(model_name, device=device)
    
    embeddings = sentence_model.encode(data, convert_to_tensor=True, device=device)
    
    return embeddings

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, model_name="all-MiniLM-L6-v2"):
        self.embeddings = embeddings
        self.model_name = model_name

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

def get_loader(embeddings, batch_size=32, shuffle=False, model_name="all-MiniLM-L6-v2"):
    dataset = EmbeddingDataset(embeddings, model_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_sentence_model(device, model_name="all-MiniLM-L6-v2"):
    """
    Get a sentence transformer model instance.
    
    Args:
        device (torch.device): Device to run the model on
        model_name (str, optional): Name of the sentence transformer model to use. 
                                  Defaults to "all-MiniLM-L6-v2"
    
    Returns:
        SentenceTransformer: The sentence transformer model instance
    """
    sentence_model = SentenceTransformer(model_name, device=device)
    return sentence_model

def CSVMerger(file_path, column_name):
    """
    Merges a specified column from a CSV file with a hard-coded merged dataset.
    
    Args:
        file_path (str): Path to the input CSV file
        column_name (str): Name of the column to retain from the input CSV
    
    Returns:
        None: The function overwrites the merged dataset file
    """
    # Load the input CSV file
    input_df = pd.read_csv(file_path)
    
    # Verify the column exists
    if column_name not in input_df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file")
    
    # Retain only the specified column
    input_df = input_df[[column_name]]
    
    # Load the hard-coded merged dataset
    merged_path = "advbench/Merged_Prompt_Dataset.csv"
    merged_df = pd.read_csv(merged_path)
    
    # Get the column name from the merged dataset
    merged_column_name = merged_df.columns[0]
    
    # Rename the input column to match the merged dataset
    input_df = input_df.rename(columns={column_name: merged_column_name})
    
    # Append the input data to the merged dataset
    final_df = pd.concat([merged_df, input_df], ignore_index=True)
    
    # Save the merged result back to the hard-coded path
    final_df.to_csv(merged_path, index=False)


