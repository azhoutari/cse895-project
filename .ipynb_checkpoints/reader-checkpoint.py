import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

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


