import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from reader import to_embeddings, get_loader
from autoencoder import AutoEncoder



def train(model, loader, device, save_path="checkpoint.pth", num_epochs=30):
    losses = []
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch in loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch)
            
            loss = criterion(outputs, batch)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch.size(0)
            
        epoch_loss = running_loss / len(loader.dataset)

        losses.append(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save to original path for backward compatibility
    torch.save(model.state_dict(), save_path)
    
    # Get embedding dimension from model's input dimension
    embedding_dim = model.encoder[0].in_features
    
    # Get model name from the dataset with fallback
    model_name = getattr(loader.dataset, 'model_name', "all-MiniLM-L6-v2")
    clean_model_name = model_name.replace("/", "_").replace("-", "_")
    
    # Create dynamic save path using both model name and embedding dimension
    dynamic_save_path = f"model_{clean_model_name}_AE{embedding_dim}.pth"
    torch.save(model.state_dict(), dynamic_save_path)
    print(f"Model also saved to: {dynamic_save_path}")
    
    # Save training loss plot with same naming convention
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig(f'training_loss_{clean_model_name}_AE{embedding_dim}.png')
    plt.close()

    return losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    X_train = pd.read_csv("./data/train.csv")['prompt'].tolist()

    # Set model name once
    model_name = "all-MiniLM-L6-v2"
    train_embeddings = to_embeddings(X_train, device, model_name=model_name)
    train_loader = get_loader(train_embeddings, shuffle=True, model_name=model_name)

    input_dim = train_embeddings.shape[1]
    model = AutoEncoder(input_dim=input_dim).to(device)

    losses = train(model, train_loader, device)

    
