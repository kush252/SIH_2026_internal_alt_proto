import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Flatten
        logits = logits.flatten(1)
        targets = targets.flatten(1)
        
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(1)
        dice = (2. * intersection + self.smooth) / (probs.sum(1) + targets.sum(1) + self.smooth)
        return 1 - dice

class SemanticBoundaryLoss(nn.Module):
    """
    Standard Semantic Segmentation Loss + Boundary Loss.
    """
    def __init__(self, config):
        super().__init__()
        self.num_classes = len(config.LOSS.weights) # e.g. 1 (Building) or 3
        # Weight configurations
        self.bce_weight = getattr(config.LOSS, 'bce_weight', 1.0)
        self.dice_weight = getattr(config.LOSS, 'dice_weight', 1.0)
        self.boundary_weight = getattr(config.LOSS, 'boundary_weight', 1.0)
        
        self.dice_loss = DiceLoss()
        
    def _generate_boundary_targets(self, semantic_targets):
        """
        Dynamically generates boundaries using erosion/dilation.
        semantic_targets: [B, C, H, W]
        Returns: [B, 1, H, W] representing the boundary map.
        """
        # Collapse all classes into a single binary mask for boundary extraction
        binary = (semantic_targets.sum(dim=1, keepdim=True) > 0).float()
        
        # Morphological gradient: dilation - erosion
        kernel_size = 5
        padding = kernel_size // 2
        dilation = F.max_pool2d(binary, kernel_size, stride=1, padding=padding)
        erosion = -F.max_pool2d(-binary, kernel_size, stride=1, padding=padding)
        
        boundary = dilation - erosion
        return boundary
        
    def forward(self, preds_dict, targets_dict):
        """
        preds_dict: 
            'semantic': [B, num_classes, H, W]
            'boundary': [B, 1, H, W]
        targets_dict:
            {'building': [B, 1, H, W], ...}
        """
        semantic_logits = preds_dict['semantic']
        boundary_logits = preds_dict['boundary']
        
        # Assemble targets_dict into a single tensor [B, num_classes, H, W]
        task_names = list(targets_dict.keys())
        B = semantic_logits.shape[0]
        device = semantic_logits.device
        
        target_masks = []
        for task_name in task_names:
            if task_name in targets_dict:
                target_masks.append(targets_dict[task_name])
            else:
                target_masks.append(torch.zeros(B, 1, semantic_logits.shape[2], semantic_logits.shape[3], device=device))
        
        semantic_targets = torch.cat(target_masks, dim=1).float()
        
        # Semantic Loss
        loss_bce = F.binary_cross_entropy_with_logits(semantic_logits, semantic_targets)
        loss_dice = self.dice_loss(semantic_logits, semantic_targets).mean()
        
        # Boundary Loss
        boundary_targets = self._generate_boundary_targets(semantic_targets)
        loss_boundary = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
        
        total_loss = (self.bce_weight * loss_bce + 
                      self.dice_weight * loss_dice + 
                      self.boundary_weight * loss_boundary)
        
        loss_dict = {
            'bce_loss': loss_bce,
            'dice_loss': loss_dice,
            'boundary_loss': loss_boundary,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
