import torch
import torch.nn as nn
import timm

class SwinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Load Swin-T
        self.encoder = timm.create_model(
            config.MODEL.name if hasattr(config.MODEL, 'name') else config.MODEL.encoder.name,
            pretrained=config.MODEL.pretrained if hasattr(config.MODEL, 'pretrained') else config.MODEL.encoder.pretrained,
            num_classes=0, # Remove classification head
            img_size=config.DATA.image_size
        )
        
        # SimMIM learnable mask token applied at the embedding level
        embed_dim = config.MODEL.embed_dim if hasattr(config.MODEL, 'embed_dim') else config.MODEL.encoder.embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, mean=0., std=.02)
        
    def forward(self, x, mask=None):
        """
        x: [B, C, H, W]
        mask: [B, mask_grid_size, mask_grid_size] where 1 indicates masked
        """
        # 1. Patch Embedding
        x = self.encoder.patch_embed(x)
        
        # 2. Masking
        if mask is not None:
            B, H_emb, W_emb, C = x.shape
            
            # mask is [B, 7, 7]
            mask_up = torch.nn.functional.interpolate(
                mask.unsqueeze(1).float(), 
                size=(H_emb, W_emb), 
                mode='nearest'
            ).view(B, H_emb, W_emb, 1) # [B, 56, 56, 1]
            
            # Replace masked tokens with mask_token
            x = torch.where(mask_up.bool(), self.mask_token.view(1, 1, 1, C).expand(B, H_emb, W_emb, C), x)
            
            # timm swin requires [B, H_emb * W_emb, C] or [B, H_emb, W_emb, C] depending on the version
            # The pos_drop input should match what patch_embed outputted.
            
        if hasattr(self.encoder, 'absolute_pos_embed') and self.encoder.absolute_pos_embed is not None:
            x = x + self.encoder.absolute_pos_embed
        if hasattr(self.encoder, 'pos_drop'):
            x = self.encoder.pos_drop(x)
        
        # 3. Forward through Swin blocks
        features = []
        for layer in self.encoder.layers:
            x = layer(x)
            features.append(x)
            
        # Return multiscale features
        return features
