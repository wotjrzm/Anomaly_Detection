import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Transformer VAE Model Architecture
# ==========================================
class TransformerVAE(nn.Module):
    def __init__(self, input_dim=29, embed_dim=64, hidden_dim=64, latent_dim=4, num_heads=4, num_layers=2):
        super(TransformerVAE, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.feature_embed = nn.Linear(1, embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.flatten_dim = input_dim * embed_dim
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.final_projection = nn.Linear(embed_dim, 1)

    def encode(self, x):
        # x shape: (Batch, 29) -> (Batch, 29, 1)
        x = x.unsqueeze(-1)
        
        # Embedding: (Batch, 29, 64)
        x = self.feature_embed(x) + self.pos_embed
        
        # Attention Mechanism
        x = self.transformer_encoder(x)
        
        # Flatten & Latent
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # (Batch, latent) -> (Batch, 29 * 64)
        x = self.fc_decode(z)
        
        # Reshape -> (Batch, 29, 64)
        x = x.view(-1, self.input_dim, self.embed_dim)
        
        x = self.transformer_decoder(x)
        
        x = self.final_projection(x)
        return x.squeeze(-1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

# ==========================================
# 2. Unified Loss Function (Recon + KLD + Contrastive)
# ==========================================
class LossFunction(nn.Module):
    def __init__(self, margin=1.0, contrastive_weight=0.1):
        super(LossFunction, self).__init__()
        self.margin = margin
        self.alpha = contrastive_weight

    def forward(self, recon_x, x, mu, logvar, z, labels):
        """
        Args:
            labels: 0(Normal), 1(Fraud) - Contrastive Learning의 기준
        """

        recon_loss = F.l1_loss(recon_x, x, reduction='mean')

        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        dist_sq = torch.cdist(z, z, p=2).pow(2) # (Batch, Batch)
        
        labels = labels.unsqueeze(1)
        label_match = (labels == labels.T).float() # (Batch, Batch)
        
        positive_loss = label_match * dist_sq
        
        negative_loss = (1 - label_match) * F.relu(self.margin - torch.sqrt(dist_sq + 1e-8)).pow(2)
        
        contrastive_loss = (positive_loss + negative_loss).mean()

        # Final Total Loss
        total_loss = recon_loss + (0.01 * kld_loss) + (self.alpha * contrastive_loss)
        
        return total_loss, recon_loss, kld_loss, contrastive_loss