import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, latent_dim=64, dropout_p=0.2):
        super(AutoEncoder, self).__init__()
        # Encoder: Input -> hidden1 -> hidden2 -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU()
        )
        # Decoder: latent -> hidden2 -> hidden1 -> Output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
            # No activation, as we use MSELoss
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
