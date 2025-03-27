import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

def read_data(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    df = df.drop(columns=["target"]).rename(columns={"goal": "prompt"})

    X_train, X_test = train_test_split(df['prompt'].tolist(), test_size=test_size, random_state=random_state)

    return X_train, X_test

def to_embeddings(data, device):
    model_name = "all-MiniLM-L6-v2"  # a small, fast, and effective model
    sentence_model = SentenceTransformer(model_name, device=device)
    
    def get_sentence_embedding(text):
        return sentence_model.encode(text, convert_to_tensor=True).clone().detach().to(device)

    embeddings = [get_sentence_embedding(text) for text in data]

    return torch.stack(embeddings)

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx]

def get_loader(embeddings, batch_size=32, shuffle=False):
    dataset = EmbeddingDataset(embeddings)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_sentence_model(device):
    model_name = "all-MiniLM-L6-v2"  # a small, fast, and effective model
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


