import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase1_encoder_loader import load_phase1_encoder
from .deeplabv3plus import DeepLabV3PlusDecoder

class Phase2MultiTaskModel(nn.Module):
    """
    Phase 2 Architecture: ResNet50 Encoder + DeepLabV3+ Decoder (Semantic + Boundary)
    """
    def __init__(self, config):
        super().__init__()
        
        # 1. Load Pretrained Encoder
        self.encoder = load_phase1_encoder(config)
        
        # 2. DeepLabV3+ Decoder
        self.decoder = DeepLabV3PlusDecoder(config)
        
    def forward(self, x):
        """
        x: [B, 3, H, W]
        Returns: 
            Dictionary containing:
            "semantic": [B, num_classes, H, W]
            "boundary": [B, 1, H, W]
        """
        # Get ResNet50 multi-scale features
        features = self.encoder(x)
        
        # Decode
        semantic_logits, boundary_logits = self.decoder(features)
        
        # Upsample to original resolution (DeepLab output is OS=4 due to C2 being H/4)
        semantic_logits = F.interpolate(semantic_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        boundary_logits = F.interpolate(boundary_logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return {
            "semantic": semantic_logits,
            "boundary": boundary_logits
        }

    @torch.no_grad()
    def semantic_inference(self, x, class_names):
        """
        Passthrough for the semantic logits, mapping them to class_names dict.
        """
        preds = self.forward(x)
        semantic_logits = preds['semantic']
        
        # Convert to dict format expected by metrics and visualization
        out_dict = {}
        for i, name in enumerate(class_names):
            out_dict[name] = semantic_logits[:, i:i+1] # [B, 1, H, W]
            
        return out_dict
