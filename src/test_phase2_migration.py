import torch
import os
import argparse
from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel
from losses.multitask_loss import SemanticBoundaryLoss
import time

def generate_dummy_data(batch_size=4, image_size=512, num_classes=1):
    images = torch.randn(batch_size, 3, image_size, image_size)
    targets = {
        'building': torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()
    }
    return images, targets

def test_initialization(config, ablation='simsiam'):
    print(f"\n[1] Testing Initialization ({ablation})...")
    
    if ablation == 'random':
        config.MODEL.encoder.pretrained = False
        config.MODEL.encoder.checkpoint_path = ""
    elif ablation == 'imagenet':
        config.MODEL.encoder.pretrained = True
        config.MODEL.encoder.checkpoint_path = ""
    elif ablation == 'simsiam':
        config.MODEL.encoder.pretrained = False
        # Uses config checkpoint path
        
    try:
        model = Phase2MultiTaskModel(config)
        print("PASS: Phase2MultiTaskModel initialized successfully.")
        
        # Test shape
        images, targets = generate_dummy_data(batch_size=2, image_size=256)
        preds = model(images)
        
        print(f"PASS: Semantic Output Shape: {preds['semantic'].shape}")
        print(f"PASS: Boundary Output Shape: {preds['boundary'].shape}")
        
        return model
    except Exception as e:
        print(f"FAIL: Initialization error: {e}")
        return None

def test_loss(model, config):
    print("\n[2] Testing SemanticBoundaryLoss...")
    criterion = SemanticBoundaryLoss(config)
    images, targets = generate_dummy_data(batch_size=2, image_size=256)
    
    preds = model(images)
    loss, loss_dict = criterion(preds, targets)
    
    print(f"PASS: Loss computed. Total: {loss.item():.4f}")
    print(f"      Loss Dict: {loss_dict}")
    
    loss.backward()
    
    has_grad = False
    for param in model.decoder.parameters():
        if param.grad is not None:
            has_grad = True
            break
            
    if has_grad:
        print("PASS: Gradient flowed back through decoder.")
    else:
        print("FAIL: No gradient in decoder.")

def test_overfit(config):
    print("\n[3] Testing Overfit (5 steps)...")
    model = Phase2MultiTaskModel(config)
    model.train()
    
    criterion = SemanticBoundaryLoss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    images, targets = generate_dummy_data(batch_size=2, image_size=256)
    
    for i in range(5):
        optimizer.zero_grad()
        preds = model(images)
        loss, _ = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        print(f"Step {i+1} Loss: {loss.item():.4f}")
        
    print("PASS: Overfit loop completed without error.")

def profile_model(config):
    print("\n[4] Profiling Model Complexity...")
    model = Phase2MultiTaskModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters (ResNet50 + DeepLabV3+): {total_params:,}")
    
    # Measure time
    model.eval()
    images, _ = generate_dummy_data(batch_size=4, image_size=512)
    with torch.no_grad():
        start = time.time()
        _ = model(images)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
    print(f"Inference time (Batch=4, Size=512): {(end - start)*1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase2_building.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # For testing without checkpoint, let's mock the checkpoint path if it doesn't exist
    if not os.path.exists(config.MODEL.encoder.checkpoint_path):
        print(f"WARNING: Checkpoint {config.MODEL.encoder.checkpoint_path} not found. Running with Random Initialization for tests.")
        config.MODEL.encoder.checkpoint_path = ""
        
    model = test_initialization(config, ablation='simsiam')
    if model:
        test_loss(model, config)
        test_overfit(config)
        profile_model(config)
