import numpy as np
import torch

class TargetBuilder:
    """
    Converts RGB semantic masks from SVAMITVA into individual binary masks
    for the specific task heads (Building, Road, Water).
    """
    def __init__(self, config):
        self.classes = config.DATA.classes
        
    def build_targets(self, rgb_mask):
        """
        Args:
            rgb_mask: NumPy array of shape (H, W, 3) or (H, W, 4)
            
        Returns:
            dict of binary masks (tensors) of shape (1, H, W) for each task.
        """
        # Ensure RGB
        if rgb_mask.shape[-1] == 4:
            rgb_mask = rgb_mask[..., :3]
            
        # Initialize output dictionary
        targets = {}
        
        for class_name, rgb_val in self.classes.items():
            if class_name == 'background':
                continue
                
            # Create a binary mask where the rgb matches the target color
            # Allow small tolerance for JPEG/PNG artifacts if any, though SVAMITVA pngs seem precise
            target_color = np.array(rgb_val, dtype=rgb_mask.dtype)
            
            # Boolean mask (H, W)
            mask = np.all(rgb_mask == target_color, axis=-1)
            
            # Convert to float tensor (1, H, W)
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
            targets[class_name] = mask_tensor
            
        return targets
