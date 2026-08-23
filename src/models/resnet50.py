import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        pretrained = config.MODEL.encoder.pretrained
        
        # Support output stride 16 or 8 for DeepLabV3+
        output_stride = getattr(config.MODEL, 'output_stride', 32)
        replace_stride_with_dilation = None
        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.encoder = resnet50(weights=weights, replace_stride_with_dilation=replace_stride_with_dilation)
        else:
            self.encoder = resnet50(weights=None, replace_stride_with_dilation=replace_stride_with_dilation)
            
        in_channels = getattr(config.DATA, 'in_channels', 3)
        if in_channels != 3:
            # Adjust the first convolution layer to accept non-RGB inputs
            original_conv1 = self.encoder.conv1
            self.encoder.conv1 = nn.Conv2d(
                in_channels, original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            # We initialize the new conv1 weights by duplicating/averaging the original 3 channels if pretrained
            if pretrained:
                with torch.no_grad():
                    # Average the 3 channel weights and repeat for in_channels
                    weight_avg = original_conv1.weight.mean(dim=1, keepdim=True)
                    self.encoder.conv1.weight.data = weight_avg.repeat(1, in_channels, 1, 1)

        # Output stride can be customized if needed, but standard ResNet is 32.
        
        # We don't need the classification head (fc)
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        """
        Forward pass returning multi-scale features for downstream tasks.
        x: [B, C, H, W]
        Returns:
            dict with keys:
                - 'c2': [B, 256, H/4, W/4]
                - 'c3': [B, 512, H/8, W/8]
                - 'c4': [B, 1024, H/16, W/16]
                - 'c5': [B, 2048, H/32, W/32]
        """
        features = {}
        
        # Stem
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        # Residual blocks
        x = self.encoder.layer1(x)
        features['c2'] = x
        
        x = self.encoder.layer2(x)
        features['c3'] = x
        
        x = self.encoder.layer3(x)
        features['c4'] = x
        
        x = self.encoder.layer4(x)
        features['c5'] = x
        
        return features
