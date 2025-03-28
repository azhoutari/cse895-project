import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from reader import to_embeddings, get_loader
from autoencoder import AutoEncoder



def train(model, loader, device, save_path="checkpoint.pth", num_epochs=30, model_name="all-MiniLM-L6-v2", embedding_dim=None):
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
    
    # Save to new dynamic path if embedding_dim is provided
    if embedding_dim is not None:
        # Create a clean model name for the file path
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        dynamic_save_path = f"model_{clean_model_name}_AE{embedding_dim}.pth"
        torch.save(model.state_dict(), dynamic_save_path)
        print(f"Model also saved to: {dynamic_save_path}")

    return losses

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = pd.read_csv("./data/train.csv")['prompt'].tolist()

    train_embeddings = to_embeddings(X_train, device)

    train_loader = get_loader(train_embeddings, shuffle=True)

    input_dim = train_embeddings.shape[1]
    model = AutoEncoder(input_dim=input_dim).to(device)

    losses = train(model, train_loader, device, embedding_dim=input_dim)

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig('training_loss.png')
    plt.close()

    
