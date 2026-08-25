import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import unittest.mock as mock
import matplotlib.patches as mpatches

# Adjust imports based on your Kaggle notebook structure
from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel

def load_and_infer(model_path, config_path, img_tensor, map_location):
    """Helper to load a specific model and get its logits for the image."""
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    print(f"Initializing model for {config_path}...")
    with mock.patch('torch.load', return_value={}):
        model = Phase2MultiTaskModel(config)
    
    checkpoint = torch.load(model_path, map_location=map_location)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(map_location)
    model.eval()
    
    print("Running inference...")
    with torch.no_grad():
        with torch.autocast(device_type=map_location.type, dtype=torch.float16, enabled=config.TRAINING.use_amp):
            preds = model(img_tensor)
            
    # Get the logits for the first (primary) class
    semantic_logits = preds['semantic'][:, 0:1] 
    return semantic_logits

def visualize_combined_prediction(image_path, bldg_model_path, bldg_config_path, road_model_path, road_config_path, device="cuda"):
    """
    Runs both the Building and Road models on the same image.
    Resolves overlaps by picking the prediction with higher confidence.
    """
    map_location = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading test image: {os.path.basename(image_path)}")
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size # (W, H)
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(map_location)
    
    # 1. Get Building Predictions
    print("--- Processing Building Model ---")
    bldg_logits = load_and_infer(bldg_model_path, bldg_config_path, img_tensor, map_location)
    
    # 2. Get Road Predictions
    print("--- Processing Road Model ---")
    road_logits = load_and_infer(road_model_path, road_config_path, img_tensor, map_location)
    
    # 3. Upsample back to original image size
    bldg_logits = torch.nn.functional.interpolate(
        bldg_logits, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False
    )
    road_logits = torch.nn.functional.interpolate(
        road_logits, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False
    )
    
    # 4. Convert logits to probabilities
    prob_b = torch.sigmoid(bldg_logits).cpu().squeeze().numpy()
    prob_r = torch.sigmoid(road_logits).cpu().squeeze().numpy()
    
    # 5. Resolve overlaps: Keep the class with the highest probability
    # Threshold at > 0.5 first
    valid_b = prob_b > 0.5
    valid_r = prob_r > 0.5
    
    # Final masks
    mask_b = valid_b & (prob_b > prob_r)
    mask_r = valid_r & (prob_r >= prob_b)
    
    # 6. Visualization
    img_np = np.array(img)
    overlay = img_np.copy()
    
    # Buildings -> Red, Roads -> Blue
    overlay[mask_b] = overlay[mask_b] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay[mask_r] = overlay[mask_r] * 0.5 + np.array([0, 0, 255]) * 0.5
    
    # Create a combined categorical mask for display
    combined_mask = np.zeros_like(prob_b)
    combined_mask[mask_r] = 1 # Road is 1
    combined_mask[mask_b] = 2 # Building is 2
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    axes[0].imshow(img_np)
    axes[0].set_title("1. Original Image", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Custom color map for the mask: 0=Black, 1=Blue (Road), 2=Red (Building)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['black', 'blue', 'red'])
    axes[1].imshow(combined_mask, cmap=cmap, interpolation='nearest')
    axes[1].set_title("2. Combined Network Output", fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("3. Final Resolved Overlay", fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    # Add a legend
    red_patch = mpatches.Patch(color='red', label='Building')
    blue_patch = mpatches.Patch(color='blue', label='Road')
    plt.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# USAGE IN KAGGLE NOTEBOOK
# ==========================================
if __name__ == "__main__":
    IMAGE_PATH = "/kaggle/input/datasets/kushhhhhh16/svamitva-dataset/kaggle_svamitva/Svamitva/FilteredData/Images/some_test_image.png"
    
    BLDG_MODEL = "/kaggle/working/outputs_phase2_building/phase2_best.pt"
    BLDG_CONFIG = "src/configs/phase2_building.yaml"
    
    ROAD_MODEL = "/kaggle/working/outputs_phase2_road/phase2_best.pt"
    ROAD_CONFIG = "src/configs/phase2_road.yaml"
    
    # visualize_combined_prediction(IMAGE_PATH, BLDG_MODEL, BLDG_CONFIG, ROAD_MODEL, ROAD_CONFIG)
