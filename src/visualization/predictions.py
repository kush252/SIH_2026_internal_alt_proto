import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision.transforms.functional as TF

def overlay_mask(image, mask, color, alpha=0.5):
    """
    Overlays a binary mask onto an RGB image with a specific color.
    image: (H, W, 3) uint8 numpy array
    mask: (H, W) boolean or 0/1 numpy array
    color: tuple (R, G, B)
    """
    colored_mask = np.zeros_like(image, dtype=np.float32)
    colored_mask[:] = color
    
    mask_bool = mask > 0.5
    
    output = image.astype(np.float32).copy()
    output[mask_bool] = output[mask_bool] * (1 - alpha) + colored_mask[mask_bool] * alpha
    
    return np.clip(output, 0, 255).astype(np.uint8)

def save_prediction_visualizations(image_tensor, preds_dict, targets_dict, config, save_dir, filename, threshold=0.5):
    """
    Saves visual comparisons of GT vs Prediction for each task.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert image to numpy (H, W, 3)
    img_np = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    classes = config.DATA.classes
    
    fig, axes = plt.subplots(len(preds_dict), 3, figsize=(15, 5 * len(preds_dict)))
    if len(preds_dict) == 1:
        axes = [axes]
        
    for i, task in enumerate(preds_dict.keys()):
        color = classes.get(task, [255, 255, 255])
        
        gt_mask = targets_dict[task].cpu().squeeze().numpy()
        pred_prob = torch.sigmoid(preds_dict[task]).cpu().squeeze().numpy()
        
        # Resize pred to match image size if needed
        if pred_prob.shape != img_np.shape[:2]:
            import cv2
            pred_prob = cv2.resize(pred_prob, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            
        pred_mask = pred_prob > threshold
        
        gt_overlay = overlay_mask(img_np, gt_mask, color)
        pred_overlay = overlay_mask(img_np, pred_mask, color)
        
        ax_gt = axes[i][0]
        ax_pred = axes[i][1]
        ax_err = axes[i][2]
        
        ax_gt.imshow(gt_overlay)
        ax_gt.set_title(f"GT: {task}")
        ax_gt.axis('off')
        
        ax_pred.imshow(pred_overlay)
        ax_pred.set_title(f"Pred: {task}")
        ax_pred.axis('off')
        
        # Error map: FP=Red, FN=Blue, TP=Green, TN=Black
        err_map = np.zeros_like(img_np)
        err_map[(pred_mask == 1) & (gt_mask == 0)] = [255, 0, 0] # FP
        err_map[(pred_mask == 0) & (gt_mask == 1)] = [0, 0, 255] # FN
        err_map[(pred_mask == 1) & (gt_mask == 1)] = [0, 255, 0] # TP
        
        ax_err.imshow(err_map)
        ax_err.set_title("Errors (Red=FP, Blue=FN, Green=TP)")
        ax_err.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_{filename}"))
    plt.close()
