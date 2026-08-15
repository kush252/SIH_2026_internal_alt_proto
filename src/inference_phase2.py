import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import cv2

from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel
from postprocessing.polygonize import mask_to_polygons
from visualization.predictions import overlay_mask

def predict(image_path, model, config, device, out_dir):
    model.eval()
    
    filename = os.path.basename(image_path)
    img = Image.open(image_path).convert('RGB')
    orig_size = img.size # (W, H)
    
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # AMP for inference
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.TRAINING.use_amp):
            preds = model(img_tensor)
            
    img_np = np.array(img)
    class_names = list(config.LOSS.weights.keys())
    
    # Process outputs
    semantic_logits = preds['semantic']
    for i, task in enumerate(class_names):
        logits = semantic_logits[:, i:i+1]
        
        # Upsample to orig size just in case
        logits = torch.nn.functional.interpolate(logits, size=(orig_size[1], orig_size[0]), mode='bilinear', align_corners=False)
        
        prob = torch.sigmoid(logits).cpu().squeeze().numpy()
        binary_mask = prob > 0.5
        
        # Save raw mask
        mask_save_path = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_{task}_mask.png")
        cv2.imwrite(mask_save_path, (binary_mask * 255).astype(np.uint8))
        
        # Overlay
        color = config.DATA.classes.get(task, [255, 255, 255])
        overlay = overlay_mask(img_np, binary_mask, color)
        
        overlay_save_path = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_{task}_overlay.png")
        # cv2 uses BGR for imwrite, overlay_mask returns RGB
        cv2.imwrite(overlay_save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Polygonize
        polygons = mask_to_polygons(binary_mask)
        print(f"[{task}] Found {len(polygons)} polygons in {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to phase2_best.pt")
    parser.add_argument("--config", type=str, default="configs/phase2.yaml")
    parser.add_argument("--out", type=str, default="inference_output")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(config.SYSTEM.device if torch.cuda.is_available() else "cpu")
    
    print("Loading model...")
    model = Phase2MultiTaskModel(config)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    os.makedirs(args.out, exist_ok=True)
    print(f"Running inference on {args.image}...")
    predict(args.image, model, config, device, args.out)
    print(f"Results saved to {args.out}")
