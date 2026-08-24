import torch
from .resnet50 import ResNet50Encoder

def load_phase1_encoder(config):
    """
    Instantiates the ResNet50 encoder and safely loads the Phase 1 SimSiam 
    pretrained weights, checking for compatibility.
    """
    encoder_cfg = config.MODEL.encoder
    
    print(f"Instantiating ResNet50 Encoder: {encoder_cfg.name}...")
    encoder = ResNet50Encoder(config)
    
    checkpoint_path = encoder_cfg.checkpoint_path
    print(f"Loading Phase 1 weights from: {checkpoint_path}")
    
    try:
        if not checkpoint_path or not __import__('os').path.exists(checkpoint_path):
            print(f"WARNING: Phase 1 checkpoint not found at {checkpoint_path}. Skipping load (Expected if running Phase 2 Inference).")
            return encoder
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Handle prefix mismatch: checkpoint has 'patch_embed...' but model expects 'encoder.patch_embed...'
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('encoder.') and k != 'mask_token':
                mapped_state_dict[f'encoder.{k}'] = v
            else:
                mapped_state_dict[k] = v
                
        # Load weights with strict=True to ensure architecture matches
        missing_keys, unexpected_keys = encoder.load_state_dict(mapped_state_dict, strict=False)
        
        if missing_keys:
            print(f"WARNING: Missing keys in encoder load: {missing_keys}")
            
        # The mask_token is expected to be unexpected if it was in the state dict 
        # (since we don't use it for Phase 2), so we filter that out of unexpected warnings.
        unexpected_keys = [k for k in unexpected_keys if 'mask_token' not in k]
        
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in encoder load: {unexpected_keys}")
            
        print("Phase 1 Encoder weights loaded successfully.")
        
    except Exception as e:
        print(f"WARNING: Failed to load Phase 1 checkpoint: {e}. Skipping load (Expected if running Phase 2 Inference).")
        
    return encoder
