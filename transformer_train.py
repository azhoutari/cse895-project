import os
import torch
from transformers import AdamW
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformer_autoencoder import TransformerAutoEncoder

# A simple Dataset for raw text prompts
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

def get_text_loader(texts, batch_size=8, shuffle=True):
    dataset = TextDataset(texts)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(model_ae, loader, device, num_epochs=3, save_path="transformer_checkpoint.pth"):
    losses = []
    optimizer = AdamW(model_ae.model.parameters(), lr=5e-5)
    model_ae.model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = model_ae.forward(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch)
        epoch_loss = running_loss / len(loader.dataset)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    torch.save(model_ae.model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    # Save training loss plot
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Transformer Autoencoder Training Loss')
    plt.savefig('transformer_training_loss.png')
    plt.close()
    return losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/bart-base"
    max_length = 128
    
    # Load training data (assumes a CSV with a 'prompt' column)
    X_train = pd.read_csv("./data/train.csv")['prompt'].tolist()
    
    # Create DataLoader for text data
    train_loader = get_text_loader(X_train, batch_size=64, shuffle=True)
    
    # Initialize the transformer autoencoder
    transformer_ae = TransformerAutoEncoder(model_name=model_name, device=device, max_length=max_length)
    
    # Train the model
    losses = train(transformer_ae, train_loader, device, num_epochs=5)
