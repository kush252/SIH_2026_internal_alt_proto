import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderPixelDecoder(nn.Module):
    """
    Pixel Decoder based on Transformer + FPN (Mask2Former Baseline).
    Avoids Deformable Attention CUDA requirements while retaining True Mask2Former multiscale traits.
    """
    def __init__(self, in_channels_list, hidden_dim=256, num_encoder_layers=6):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Projections to hidden_dim
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(len(in_channels_list))
        ])
        
        self.mask_feature_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Standard Transformer Encoder for the lowest resolution (1/32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=2048, 
            dropout=0.1, 
            activation='relu', 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
    def forward(self, features):
        """
        features: [B, C1, H/4, W/4], [B, C2, H/8, W/8], [B, C3, H/16, W/16], [B, C4, H/32, W/32]
        Returns:
            mask_features: [B, hidden_dim, H/4, W/4]
            multi_scale_features: list of [B, hidden_dim, H_i, W_i]
        """
        # Project all
        lateral_features = [
            lat_conv(feat) for feat, lat_conv in zip(features, self.lateral_convs)
        ]
        
        # Apply Transformer Encoder to lowest resolution (1/32)
        coarsest = lateral_features[-1]
        B, C, H, W = coarsest.shape
        coarsest_flat = coarsest.flatten(2).permute(0, 2, 1) # [B, H*W, C]
        coarsest_encoded = self.transformer_encoder(coarsest_flat)
        coarsest = coarsest_encoded.permute(0, 2, 1).view(B, C, H, W)
        lateral_features[-1] = coarsest
        
        # FPN Top-Down Fusion
        x = lateral_features[-1]
        fused_features = [self.output_convs[-1](x)]
        
        for i in range(len(features) - 2, -1, -1):
            x = F.interpolate(x, size=lateral_features[i].shape[-2:], mode='bilinear', align_corners=False)
            x = x + lateral_features[i]
            fused_features.insert(0, self.output_convs[i](x))
            
        mask_features = self.mask_feature_proj(fused_features[0])
        return mask_features, fused_features

class MaskedAttentionDecoderLayer(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Self Attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, value, attn_mask=None):
        """
        query: [B, N, C]
        value: [B, H*W, C] (Flattened multi-scale features)
        attn_mask: [B*nheads, N, H*W] - True means DO NOT attend
        """
        # 1. Masked Cross Attention (Query attends to Image Features)
        ca_out, _ = self.cross_attn(query, value, value, attn_mask=attn_mask)
        query = self.norm1(query + ca_out)
        
        # 2. Self Attention (Queries attend to each other)
        sa_out, _ = self.self_attn(query, query, query)
        query = self.norm2(query + sa_out)
        
        # 3. FFN
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = self.norm3(query + ffn_out)
        
        return query

class TrueMask2FormerDecoder(nn.Module):
    """
    Iterative Transformer Decoder with Learnable Queries and Masked Attention.
    """
    def __init__(self, num_classes, hidden_dim=256, num_queries=100, num_layers=6, nheads=8):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        
        # Learnable Queries
        self.query_embeddings = nn.Embedding(num_queries, hidden_dim)
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            MaskedAttentionDecoderLayer(hidden_dim, nheads)
            for _ in range(num_layers)
        ])
        
        # Prediction Heads (Shared across layers)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) # +1 for background (no object)
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, multi_scale_features, mask_features):
        """
        multi_scale_features: list of [B, C, H_i, W_i] (typically 4 scales)
        mask_features: [B, C, H/4, W/4] (1/4 scale for final dot product)
        """
        B = mask_features.shape[0]
        
        # Initialize queries
        query = self.query_embeddings.weight.unsqueeze(0).repeat(B, 1, 1) # [B, N, C]
        
        outputs_class = []
        outputs_mask = []
        
        # In Mask2Former, we process multi-scale features in round-robin fashion or concat.
        # We concatenate 1/8, 1/16, 1/32 scales for cross-attention.
        src_features = []
        spatial_shapes = []
        for feat in multi_scale_features[1:]: # Skip 1/4 scale for cross-attn
            _, _, H, W = feat.shape
            spatial_shapes.append((H, W))
            src_features.append(feat.flatten(2).permute(0, 2, 1)) # [B, H*W, C]
            
        src_flattened = torch.cat(src_features, dim=1) # [B, sum(H*W), C]
        
        # Initial mask prediction (from learnable queries directly)
        mask_embed_fp32 = self.mask_embed(query).float()
        mask_features_fp32 = mask_features.float()
        mask_pred = torch.einsum("bnc,bchw->bnhw", mask_embed_fp32, mask_features_fp32)
        outputs_mask.append(mask_pred)
        outputs_class.append(self.class_embed(query).float())
        
        for i, layer in enumerate(self.layers):
            attn_masks = []
            for (H, W) in spatial_shapes:
                # Resize previous mask_pred to match spatial shape
                mask_resized = F.interpolate(mask_pred, size=(H, W), mode="bilinear", align_corners=False)
                # Binarize mask: True means ignore
                bool_mask = (mask_resized < 0).flatten(2) # [B, N, H_i*W_i]
                attn_masks.append(bool_mask)
            
            attn_mask = torch.cat(attn_masks, dim=-1) # [B, N, sum(H*W)]
            
            # Prevent all-True rows which cause NaN in softmax
            # If a query has all True (all pixels masked out), unmask everything (all False)
            all_true = attn_mask.all(dim=-1, keepdim=True)
            attn_mask = attn_mask.masked_fill(all_true, False)
            
            # PyTorch MultiheadAttention expects attn_mask of shape [B * nheads, N, S]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, layer.cross_attn.num_heads, 1, 1) # [B, nheads, N, S]
            attn_mask = attn_mask.view(-1, query.size(1), src_flattened.size(1)) # [B*nheads, N, S]
            
            # Masked Cross Attention + Self Attention + FFN
            query = layer(query, src_flattened, attn_mask=attn_mask)
            
            # Predict Class and Mask
            class_logits = self.class_embed(query).float() # [B, N, num_classes + 1]
            mask_embed_fp32 = self.mask_embed(query).float()
            mask_logits = torch.einsum("bnc,bchw->bnhw", mask_embed_fp32, mask_features_fp32) # [B, N, H/4, W/4]
            
            outputs_class.append(class_logits)
            outputs_mask.append(mask_logits)
            
            mask_pred = mask_logits # Update for next iteration
            
        return outputs_class, outputs_mask
