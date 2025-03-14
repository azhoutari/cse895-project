import os
import torch
import torch.optim as optim
import torch.nn as nn

from reader import read_data, to_embeddings, get_loader
from autoencoder import AutoEncoder

def train(model, loader, save_path="checkpoint.pth", num_epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch)
            
            loss = criterion(outputs, batch)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch.size(0)
            
        epoch_loss = running_loss / len(loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test = read_data("./advbench/harmful_behaviors.csv")

    train_embeddings = to_embeddings(X_train, device)

    train_loader = get_loader(train_embeddings, shuffle=True)

    input_dim = train_embeddings.shape[1]
    model = AutoEncoder(input_dim=input_dim).to(device)

    train(model, train_loader)

    
