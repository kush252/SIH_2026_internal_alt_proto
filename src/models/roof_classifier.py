import torch
import torch.nn as nn
from .phase1_encoder_loader import load_phase1_encoder

class Phase3RoofClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        
        # Load the Phase 1 pretrained ResNet50 encoder
        self.encoder = load_phase1_encoder(config)
        
        # ResNet50's final feature map ('c5') has 2048 channels
        encoder_out_dim = 2048
        
        # Global Average Pooling to convert spatial features to a vector
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple head per spec: Dropout (compensates for no Stochastic Depth) -> Linear
        # ResNet has no built-in Stochastic Depth unlike ConvNeXt, so we add Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(encoder_out_dim, num_classes)
        )
        
    def forward(self, x):
        # Extract features (returns a dictionary of feature maps)
        features = self.encoder(x)
        
        # Get the highest-level feature map: [B, 2048, H/32, W/32]
        c5 = features['c5']
        
        # Pool to [B, 2048, 1, 1] then flatten to [B, 2048]
        pooled = self.gap(c5)
        flattened = torch.flatten(pooled, 1)
        
        # Predict class logits
        logits = self.classifier(flattened)
        return logits

    def freeze_backbone(self):
        """Freeze the encoder; useful for a warm-up phase to train the head first."""
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        print("Encoder frozen. Only classification head will be trained.")

    def unfreeze_backbone(self):
        """Unfreeze the encoder for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen. Full fine-tuning enabled.")
