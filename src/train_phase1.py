import torch
from torch.utils.data import DataLoader
from utils.config import load_config
from datasets.transforms import SimSiamTransform
from datasets.unified_ssl_dataset import UnifiedSSLDataset
from models.simsiam import SimSiam
from training.trainer_simsiam import SimSiamTrainer
import os
import random
import numpy as np
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="SimSiam Phase 1 Training")
    parser.add_argument('--config', type=str, default=r"configs\simsiam_resnet50.yaml")
    parser.add_argument('--kaggle_dir', type=str, help="Path to Kaggle dataset (overrides config)")
    parser.add_argument('--landcover_dir', type=str, help="Path to Landcover dataset (overrides config)")
    parser.add_argument('--output_dir', type=str, help="Path to save outputs (overrides config)")
    parser.add_argument('--resume', type=str, help="Path to checkpoint .pt file to resume training")
    parser.add_argument('--batch_size', type=int, help="Override batch_size in config")
    parser.add_argument('--use_amp', type=lambda x: (str(x).lower() == 'true'), help="Override use_amp in config (True/False)")
    parser.add_argument('--max_steps', type=int, default=None, help="Stop training early after this many steps")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.kaggle_dir:
        config.DATA.datasets.kaggle.path = args.kaggle_dir
    if args.landcover_dir:
        config.DATA.datasets.landcover.path = args.landcover_dir
    if args.output_dir:
        config.SYSTEM.output_dir = args.output_dir
    if args.batch_size:
        config.TRAINING.batch_size = args.batch_size
    if args.use_amp is not None:
        config.TRAINING.use_amp = args.use_amp
    
    set_seed(config.SYSTEM.seed)
    
    print("Setting up datasets...")
    transform = SimSiamTransform(config)
    dataset = UnifiedSSLDataset(config, transform=transform)
    
    # Use the full dataset for production (num_samples=None uses all patches)
    sampler = dataset.get_sampler(num_samples=None)
    
    # Custom collate_fn because UnifiedSSLDataset expects dictionary from dataset
    def simsiam_collate(batch):
        # batch is a list of dicts {"image": view1, "mask": view2}? No, wait!
        # UnifiedSSLDataset doesn't change the return type of the dataset.
        # Let's check what datasets return.
        # Wait, the transforms for SSL are applied to PIL image.
        # The KaggleAerialDataset does:
        # if self.transform: image = self.transform(image)
        # return {"image": image}
        # If transform returns a tuple (view1, view2), then image is a tuple.
        # So batch is [{"image": (view1_a, view2_a)}, {"image": (view1_b, view2_b)}]
        view1_batch = []
        view2_batch = []
        for item in batch:
            # The datasets (KaggleAerial/Landcover) assume transform returns (image, mask).
            # For SimSiam, it returns (view1, view2). So view1 is in 'image' and view2 is in 'mask'.
            view1 = item["image"]
            view2 = item["mask"]
            view1_batch.append(view1)
            view2_batch.append(view2)
        return torch.stack(view1_batch), torch.stack(view2_batch)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.TRAINING.batch_size,
        sampler=sampler,
        num_workers=config.TRAINING.num_workers,
        pin_memory=True,
        collate_fn=simsiam_collate
    )
    
    print("Initializing Model...")
    model = SimSiam(config)
    
    trainer = SimSiamTrainer(
        model=model,
        dataloader=dataloader,
        config=config,
        output_dir=config.SYSTEM.output_dir,
        resume_path=args.resume
    )
    
    try:
        trainer.train(max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user!")
        
    print("Rescuing encoder weights directly from memory...")
    import os
    os.makedirs(config.SYSTEM.output_dir, exist_ok=True)
    rescue_path = os.path.join(config.SYSTEM.output_dir, "phase1_simsiam_encoder.pt")
    
    encoder_state = {}
    state_dict = trainer.model.state_dict()
    for k, v in state_dict.items():
        if 'encoder.' in k:
            clean_key = k.split('encoder.')[-1]
            encoder_state[clean_key] = v
            
    torch.save(encoder_state, rescue_path)
    print(f"Successfully rescued {len(encoder_state)} tensors and saved to {rescue_path}!")
    print("You can now safely proceed to Phase 2!")

if __name__ == "__main__":
    main()
