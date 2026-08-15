import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel

def load_model(config_path, device):
    """Loads a specific model based on its configuration file."""
    config = load_config(config_path)
    model = Phase2MultiTaskModel(config)
    
    # The best weights are saved in the output_dir
    weights_path = os.path.join("src", config.SYSTEM.output_dir, "best.pt")
    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found. Returning randomly initialized model.")
        return model, config
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, config

def main():
    parser = argparse.ArgumentParser(description="Hackathon Demo: Ensemble Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="ensemble_output.png", help="Path to save output")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load all three models
    print("Loading specialized models...")
    building_model, building_cfg = load_model("src/configs/phase2_building.yaml", device)
    road_model, road_cfg = load_model("src/configs/phase2_road.yaml", device)
    water_model, water_cfg = load_model("src/configs/phase2_water.yaml", device)
    
    # Prepare image
    img_pil = Image.open(args.image).convert('RGB')
    
    # We will resize to the training size (512) for inference to match what models expect
    img_tensor = TF.resize(img_pil, [512, 512], interpolation=TF.InterpolationMode.BILINEAR)
    img_tensor = TF.to_tensor(img_tensor).unsqueeze(0).to(device) # [1, 3, 512, 512]
    
    print("Running Ensemble Inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            # Run inference using the dual-head semantic output
            b_preds = building_model(img_tensor)
            r_preds = road_model(img_tensor)
            w_preds = water_model(img_tensor)
            
    # Extract binary masks (threshold at 0.5)
    b_mask = (torch.sigmoid(b_preds['semantic'])[0, 0] > 0.5).cpu().numpy()
    r_mask = (torch.sigmoid(r_preds['semantic'])[0, 0] > 0.5).cpu().numpy()
    w_mask = (torch.sigmoid(w_preds['semantic'])[0, 0] > 0.5).cpu().numpy()
    
    # Create a composite image
    # We will overlay colors: Building=Red, Road=Yellow, Water=Blue
    print("Generating Composite Output...")
    original_resized = TF.resize(img_pil, [512, 512], interpolation=TF.InterpolationMode.BILINEAR)
    composite = np.array(original_resized).astype(np.float32) / 255.0
    
    # Create colored masks
    overlay = np.zeros_like(composite)
    overlay[b_mask] = [1.0, 0.0, 0.0]  # Red
    overlay[r_mask] = [1.0, 1.0, 0.0]  # Yellow
    overlay[w_mask] = [0.0, 0.4, 1.0]  # Blue
    
    # Combine with alpha blending
    alpha = 0.5
    mask_exists = b_mask | r_mask | w_mask
    composite[mask_exists] = composite[mask_exists] * (1 - alpha) + overlay[mask_exists] * alpha
    
    # Plot side by side
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_resized)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Ensemble DeepLabV3+ Predictions")
    plt.imshow(composite)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved ensemble prediction to {args.output}")

if __name__ == "__main__":
    main()
