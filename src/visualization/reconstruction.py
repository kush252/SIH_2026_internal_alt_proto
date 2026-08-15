import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalize ImageNet normalized tensor"""
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

def save_reconstruction_grid(original, masked_image, reconstructed, mask, epoch, step, output_dir, max_images=8):
    """
    Saves a grid showing: Original | Masked | Reconstructed | Error
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n = min(max_images, original.shape[0])
    
    orig = denormalize(original[:n].detach().cpu())
    recon = denormalize(reconstructed[:n].detach().cpu())
    m = mask[:n].float().cpu()
    
    # Masked image: apply the mask to the original image (e.g. gray out masked areas)
    gray_value = 0.5
    masked_img = orig * (1 - m) + m * gray_value
    
    # Error map
    error = torch.abs(orig - recon).mean(dim=1, keepdim=True)
    # Convert error to heatmap-like RGB by repeating channels
    error_rgb = error.repeat(1, 3, 1, 1)
    # Normalize error for better visualization
    error_rgb = error_rgb / (error_rgb.max() + 1e-5)
    
    # Concatenate side by side
    grid = torch.cat([orig, masked_img, recon, error_rgb], dim=0)
    
    # We want 4 columns (Orig, Masked, Recon, Error)
    grid_img = torchvision.utils.make_grid(grid, nrow=n, padding=2, normalize=False)
    
    plt.figure(figsize=(15, int(15 * 4 / n)))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    
    # Add titles manually (approximate)
    plt.title("Original (Top) | Masked | Reconstructed | Absolute Error (Bottom)")
    
    save_path = Path(output_dir) / f"recon_epoch_{epoch:03d}_step_{step:05d}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
