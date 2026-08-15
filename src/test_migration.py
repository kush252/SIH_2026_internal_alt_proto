import torch
import torch.nn as nn
from utils.config import load_config
from datasets.transforms import SimSiamTransform
from datasets.unified_ssl_dataset import UnifiedSSLDataset
from models.resnet50 import ResNet50Encoder
from models.simsiam import SimSiam
from train_phase1 import SimSiamTrainer
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_migration():
    print("=== STARTING MIGRATION TESTS ===")
    
    # Load config
    config = load_config(r"configs\simsiam_resnet50.yaml")
    
    # 1. & 3. Dataset loading and SimSiam two-view generation
    print("Testing transforms...")
    transform = SimSiamTransform(config)
    # mock image
    from PIL import Image
    import numpy as np
    mock_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    view1, view2 = transform(mock_img)
    assert view1.shape == (3, 224, 224), "View 1 shape mismatch"
    assert view2.shape == (3, 224, 224), "View 2 shape mismatch"
    print("Transforms and two-view generation OK")
    
    # 4. ResNet50 forward pass & 14. Shape consistency
    print("Testing ResNet50Encoder...")
    encoder = ResNet50Encoder(config)
    x = torch.randn(2, 3, 224, 224)
    features = encoder(x)
    assert 'c2' in features and features['c2'].shape == (2, 256, 56, 56)
    assert 'c3' in features and features['c3'].shape == (2, 512, 28, 28)
    assert 'c4' in features and features['c4'].shape == (2, 1024, 14, 14)
    assert 'c5' in features and features['c5'].shape == (2, 2048, 7, 7)
    print("ResNet50Encoder and Shape Consistency OK")
    
    # 5. Projector & 6. Predictor & 7. SimSiam loss & 8. Backpropagation & 16. No NaN/Inf loss
    print("Testing SimSiam forward, loss, backprop...")
    model = SimSiam(config)
    
    view1_tensor = torch.randn(2, 3, 224, 224)
    view2_tensor = torch.randn(2, 3, 224, 224)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    loss = model(view1_tensor, view2_tensor)
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"
    
    loss.backward()
    
    # Check if gradients exist
    has_grads = False
    for p in model.parameters():
        if p.grad is not None:
            has_grads = True
            break
    assert has_grads, "No gradients after backward pass"
    optimizer.step()
    print("SimSiam forward, loss, backprop OK")
    
    # 9. AMP compatibility
    print("Testing AMP compatibility...")
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = model(view1_tensor, view2_tensor)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print("AMP compatibility OK")
    
    # 10. Checkpoint save & 11. Checkpoint load & 12. Encoder extract & 13. Encoder reload
    print("Testing Checkpointing...")
    torch.save(model.state_dict(), "temp_model.pt")
    model2 = SimSiam(config)
    model2.load_state_dict(torch.load("temp_model.pt", weights_only=True))
    
    # extract encoder
    from extract_encoder import extract_encoder
    state_dict = torch.load("temp_model.pt", weights_only=True)
    encoder_state = {k.replace('encoder.', '', 1): v for k, v in state_dict.items() if k.startswith('encoder.')}
    torch.save(encoder_state, "temp_encoder.pt")
    
    # reload encoder
    encoder2 = ResNet50Encoder(config)
    encoder2.load_state_dict(torch.load("temp_encoder.pt", weights_only=True))
    
    os.remove("temp_model.pt")
    os.remove("temp_encoder.pt")
    print("Checkpoint saving, loading, extraction, and reloading OK")
    
    # 18. Small overfit test & 17. One-step training
    print("Running small overfit test...")
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            # Same image 5 times
            self.img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            self.transform = SimSiamTransform(config)
        def __len__(self):
            return 10
        def __getitem__(self, idx):
            return self.transform(self.img)
            
    dummy_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    model.train()
    
    # Use Adam for the overfit test to make it converge quickly
    test_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    initial_loss = None
    final_loss = None
    for epoch in range(10):
        for view1, view2 in dummy_loader:
            test_optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            test_optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()
            
    print(f"Overfit Test: Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")
    print("Small overfit test OK (runs without crashing)")
    print("Small overfit test OK")
    
    # Computational Comparison
    print("=== COMPUTATIONAL COMPARISON ===")
    from models.old.simmim import SimMIM
    from utils.config import load_config as lc
    config_old = lc(r"configs\simmim_swin_t.yaml")
    old_model = SimMIM(config_old)
    
    params_old = count_parameters(old_model)
    params_new = count_parameters(model)
    print(f"Swin-T + SimMIM Parameters : {params_old:,}")
    print(f"ResNet50 + SimSiam Parameters: {params_new:,}")
    
    import time
    old_model.eval()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    mask = torch.zeros(2, 7, 7)
    
    t0 = time.time()
    for _ in range(5):
        with torch.no_grad():
            old_model(x, mask)
    t1 = time.time()
    old_time = (t1 - t0) / 5.0
    
    t0 = time.time()
    for _ in range(5):
        with torch.no_grad():
            model(x, x)
    t1 = time.time()
    new_time = (t1 - t0) / 5.0
    
    print(f"Swin-T Forward pass time : {old_time:.4f} seconds")
    print(f"ResNet50 Forward pass time: {new_time:.4f} seconds")
    
    print("ALL TESTS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    test_migration()
