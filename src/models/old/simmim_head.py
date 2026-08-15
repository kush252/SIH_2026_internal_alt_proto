import torch
import torch.nn as nn

class SimMIMHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=3, patch_size=4):
        super().__init__()
        
        self.patch_size = patch_size
        self.out_dim = out_dim
        
        # A simple linear projection back to patch pixel values
        # Swin-T final dim is 768. We want to predict 4x4 patches of 3 channels = 48
        self.projection = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * patch_size * patch_size, kernel_size=1)
        )
        
    def forward(self, x):
        """
        x: [B, H, W, C] from Swin Encoder
        returns: reconstructed image [B, C, H, W]
        """
        B, H, W, C = x.shape
        
        # Reshape to [B, C, H, W] for the Conv2d
        x = x.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        
        x = self.projection(x) # [B, 48, 56, 56]
        
        # Pixel shuffle to reshape from [B, 3*16, 56, 56] to [B, 3, 224, 224]
        x = nn.functional.pixel_shuffle(x, self.patch_size)
        
        return x
