import torch
import argparse
import os

def extract_encoder(best_pt_path, output_path):
    print(f"Loading {best_pt_path}...")
    checkpoint = torch.load(best_pt_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    print("Extracting encoder weights...")
    encoder_state = {}
    
    # ResNet50 encoder keys might start with encoder. or we might need to handle Swin old format
    # SimSiam structure: self.encoder = ResNet50Encoder
    # ResNet50Encoder structure: self.encoder = resnet50
    # So weights might be encoder.encoder.conv1.weight. 
    # Let's save the whole ResNet50Encoder state so Phase 2 can load it.
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            # Keep 'encoder.' prefix or remove it?
            # Phase 2 phase1_encoder_loader expects:
            # model.load_state_dict(mapped_state_dict)
            # If we extract and save just the encoder part, 
            # we strip the first 'encoder.' prefix so it can be loaded directly into ResNet50Encoder.
            encoder_state[k.replace('encoder.', '', 1)] = v
            
    print(f"Found {len(encoder_state)} encoder tensors.")
    
    torch.save(encoder_state, output_path)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="best.pt", help="Path to best.pt")
    parser.add_argument("--output", type=str, default="swin_t_simmim_encoder.pt", help="Path to save output")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Could not find {args.input}. Please provide the correct path.")
    else:
        extract_encoder(args.input, args.output)
