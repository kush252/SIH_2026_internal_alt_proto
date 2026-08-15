import torch
import torch.nn as nn
from .swin_encoder import SwinEncoder
from .simmim_head import SimMIMHead

class SimMIM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = SwinEncoder(config)
        
        # Swin-T has 768 output dim.
        self.head = SimMIMHead(
            in_dim=self.encoder.encoder.num_features,
            out_dim=config.DATA.in_channels,
            patch_size=config.SIMMIM.mask_patch_size # 32x upsample
        )
        
        self.mask_patch_size = config.SIMMIM.mask_patch_size
        
    def forward(self, x, mask):
        """
        x: [B, C, H, W]
        mask: [B, H_mask, W_mask] where 1 is masked, 0 is visible
        """
        # Forward encoder
        z = self.encoder(x, mask)
        
        # Forward decoder
        x_rec = self.head(z[-1])
        
        # Compute L1 Loss only on masked patches
        # 1. Upsample mask to image resolution
        B, C, H, W = x.shape
        mask_up = torch.nn.functional.interpolate(
            mask.unsqueeze(1).float(),
            size=(H, W),
            mode='nearest'
        )
        
        # 2. Compute L1 loss
        loss_recon = nn.functional.l1_loss(x, x_rec, reduction='none')
        
        # 3. Apply mask and compute mean over masked pixels
        loss = (loss_recon * mask_up).sum() / (mask_up.sum() + 1e-5) / C
        
        return loss, x_rec, mask_up
