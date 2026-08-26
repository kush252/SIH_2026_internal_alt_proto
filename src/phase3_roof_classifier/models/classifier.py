import torch
import torch.nn as nn
import timm

class RoofClassifier(nn.Module):
    def __init__(self, model_name='convnext_tiny', num_classes=4, pretrained=True):
        super(RoofClassifier, self).__init__()
        # Load ConvNeXt Tiny from timm with stochastic depth (Drop Path) to fight overfitting
        self.backbone = timm.create_model(model_name, pretrained=pretrained, drop_path_rate=0.2)
        
        # Replace the classifier head
        self.backbone.reset_classifier(num_classes)
        
    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freezes all layers except the final classification head."""
        for name, param in self.backbone.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreezes all layers."""
        for param in self.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    # Quick test
    model = RoofClassifier()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)
