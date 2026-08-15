import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
        self.aspp_blocks = nn.ModuleList()
        
        # 1x1 Conv
        self.aspp_blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        
        # Atrous Convs
        for dilation in dilations:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # Global Avg Pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(len(self.aspp_blocks) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        size = x.shape[-2:]
        out = []
        for block in self.aspp_blocks:
            out.append(block(x))
            
        pool_out = self.global_pool(x)
        pool_out = F.interpolate(pool_out, size=size, mode='bilinear', align_corners=False)
        out.append(pool_out)
        
        res = torch.cat(out, dim=1)
        return self.project(res)


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ResNet50 c5 channels = 2048, c2 channels = 256
        high_level_channels = 2048
        low_level_channels = 256
        
        aspp_channels = 256
        
        output_stride = getattr(config.MODEL, 'output_stride', 16)
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            dilations = [6, 12, 18] # fallback
            
        self.aspp = ASPPModule(high_level_channels, aspp_channels, dilations)
        
        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder_blocks = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        num_classes = len(config.LOSS.weights)
        # Semantic Head (num_classes + 1 for background) - wait, if num_classes is 3 (building, road, water) + background = 4
        # We'll just use num_classes. If the config has background in LOSS.weights, num_classes is total.
        
        self.semantic_head = nn.Conv2d(256, num_classes, 1)
        
        # Boundary Head (binary for all foreground object boundaries, or per-class boundaries)
        # Let's predict a single global boundary map
        self.boundary_head = nn.Conv2d(256, 1, 1)
        
    def forward(self, features):
        """
        features is a dict from ResNet50Encoder with c2, c3, c4, c5
        """
        c2 = features['c2']
        c5 = features['c5']
        
        # 1. High-level path
        aspp_out = self.aspp(c5)
        
        # 2. Low-level path
        low_level = self.low_level_project(c2)
        
        # 3. Upsample ASPP and concat
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[-2:], mode='bilinear', align_corners=False)
        concat_feat = torch.cat([aspp_up, low_level], dim=1)
        
        # 4. Decode
        decoder_feat = self.decoder_blocks(concat_feat)
        
        # 5. Predict
        semantic_logits = self.semantic_head(decoder_feat)
        boundary_logits = self.boundary_head(decoder_feat)
        
        return semantic_logits, boundary_logits
