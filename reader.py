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
        return sentence_model.encode(prompt, convert_to_tensor=True).clone().detach().to(device)

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




