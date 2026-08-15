import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet50 import ResNet50Encoder

class SimSiam(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ResNet50Encoder(config)
        
        # For ResNet50, c5 is 2048 dimensions
        encoder_out_dim = 2048 
        projector_dim = config.SSL.projector_dim
        projector_hidden_dim = config.SSL.projector_hidden_dim
        predictor_dim = config.SSL.predictor_dim
        
        # Projector: 3-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(encoder_out_dim, projector_hidden_dim, bias=False),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_hidden_dim, bias=False),
            nn.BatchNorm1d(projector_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_dim, projector_dim, bias=False),
            nn.BatchNorm1d(projector_dim, affine=False)
        )
        
        # Predictor: 2-layer MLP
        self.predictor = nn.Sequential(
            nn.Linear(projector_dim, predictor_dim, bias=False),
            nn.BatchNorm1d(predictor_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_dim, projector_dim)
        )

    def forward_encoder_projector(self, x):
        features = self.encoder(x)
        # Use c5 (highest level features) for SSL
        c5 = features['c5']
        
        # Global Average Pooling
        c5 = c5.mean(dim=[2, 3])
        
        # Projector
        z = self.projector(c5)
        return z

    def forward(self, x1, x2):
        """
        x1: view 1 [B, C, H, W]
        x2: view 2 [B, C, H, W]
        """
        # Encoder + Projector for both views
        z1 = self.forward_encoder_projector(x1)
        z2 = self.forward_encoder_projector(x2)
        
        # Predictor for both views
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        # Negative cosine similarity loss with stop gradient
        loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
        
        return loss
        
    def criterion(self, p, z):
        """
        Cosine similarity between normalized vectors
        """
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return (p * z).sum(dim=1)
