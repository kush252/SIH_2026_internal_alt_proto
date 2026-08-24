import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Adjust imports based on your Kaggle notebook structure
from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel

def visualize_prediction(image_path, model_path, config_path, device="cuda"):
    """
    Runs inference on a single image and plots the original image, 
    the predicted mask, and the overlay side-by-side.
    """
    # 1. Load Configuration and Model
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # 2. Initialize Model and Load Weights
    print("Initializing model and loading weights...")
    model = Phase2MultiTaskModel(config)
    
    # Handle CPU vs CUDA loading
    map_location = torch.device(device)
    checkpoint = torch.load(model_path, map_location=map_location)
    
    # If saved as an optimizer dictionary vs direct state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # 3. Load and Preprocess Image
    print(f"Loading image {image_path}...")
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size # (W, H)
    
    # Preprocess (Convert to Tensor and add batch dimension)
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    # 4. Run Inference
    print("Running inference through backbone...")
    with torch.no_grad():
        # Use AMP if configured for faster inference
        with torch.autocast(device_type=device if 'cuda' in device else 'cpu', 
                            dtype=torch.float16, 
                            enabled=config.TRAINING.use_amp):
            preds = model(img_tensor)
            
    # 5. Process Output
    # Assuming single task for now (e.g., building) or taking the first task
    task_name = list(config.LOSS.weights.keys())[0] 
    
    semantic_logits = preds['semantic'][:, 0:1] # Get first task logits
    
    # Upsample back to original image size
    logits_upsampled = torch.nn.functional.interpolate(
        semantic_logits, 
        size=(orig_size[1], orig_size[0]), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Convert logits to probabilities and then binary mask
    prob = torch.sigmoid(logits_upsampled).cpu().squeeze().numpy()
    binary_mask = prob > 0.5
    
    # 6. Visualization
    img_np = np.array(img)
    
    # Create overlay (Red for building/road)
    overlay = img_np.copy()
    overlay[binary_mask] = overlay[binary_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title(f"Predicted Mask ({task_name})", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# USAGE IN KAGGLE NOTEBOOK
# ==========================================
if __name__ == "__main__":
    # Replace these paths with your actual Kaggle dataset/working paths
    IMAGE_PATH = "/kaggle/input/datasets/kushhhhhh16/svamitva-dataset/kaggle_svamitva/Svamitva/FilteredData/Images/some_test_image.png"
    MODEL_PATH = "/kaggle/working/outputs_phase2_building/phase2_best.pt"
    CONFIG_PATH = "src/configs/phase2_building.yaml"
    
    # Run the visualization
    # visualize_prediction(IMAGE_PATH, MODEL_PATH, CONFIG_PATH)
