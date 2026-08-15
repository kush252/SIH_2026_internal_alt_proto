import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import load_config
from models.task_heads import Phase2MultiTaskModel
from datasets.generic_binary_dataset import GenericBinaryDataset
from training.metrics import MultitaskMetrics

def evaluate(config_path, test_data_path, device):
    config = load_config(config_path)
    
    # 1. Load Model
    print(f"Loading model configured in {config_path}")
    model = Phase2MultiTaskModel(config)
    weights_path = os.path.join("src", config.SYSTEM.output_dir, "best.pt")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at {weights_path}. Train the model first!")
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 2. Setup Dataset
    if test_data_path:
        print(f"Loading Test Dataset from CLI override: {test_data_path}")
        config.DATA.dataset_path = test_data_path
    else:
        print(f"Loading Test Dataset from config: {config.DATA.dataset_path}")
    
    # is_train=False disables random geometric augmentations
    test_dataset = GenericBinaryDataset(config.DATA.dataset_path, config, is_train=False)
    
    # We don't apply the 90/10 split internally if we explicitly want to test on everything in the folder.
    # To enforce testing on the entire folder, we bypass the internal split of GenericBinaryDataset:
    test_dataset.image_paths = []
    test_dataset.mask_paths = []
    test_dataset._discover_files() 
    # Notice we don't call _apply_split() here, so it uses 100% of the files found in the test folder!
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.TRAINING.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Found {len(test_dataset)} images for testing.")
    
    # 3. Setup Metrics
    class_names = list(config.LOSS.weights.keys())
    metrics = MultitaskMetrics(class_names)
    
    # 4. Evaluation Loop
    print("\nStarting Evaluation...")
    pbar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=config.TRAINING.use_amp):
                # We only need the semantic inference outputs, not the loss
                semantic_preds = model.semantic_inference(images, class_names)
                
            metrics.update(semantic_preds, targets)
            
    # 5. Compute and Print Results
    results = metrics.compute()
    
    print("\n" + "="*40)
    print("FINAL TEST SET RESULTS")
    print("="*40)
    
    # Calculate Pixel Accuracy manually since MultitaskMetrics tracks TP/FP/FN/Union
    for task in class_names:
        s = metrics.stats[task]
        
        # Pixel Accuracy = (True Positives + True Negatives) / Total Pixels
        total_pixels = len(test_dataset) * config.DATA.image_size * config.DATA.image_size
        # True Negatives = Total - (TP + FP + FN)
        tn = total_pixels - (s['tp'] + s['fp'] + s['fn'])
        pixel_accuracy = (s['tp'] + tn) / total_pixels
        
        print(f"--- Task: {task.upper()} ---")
        print(f"Pixel Accuracy : {pixel_accuracy * 100:.2f}%")
        print(f"IoU Score      : {results[f'{task}_iou'] * 100:.2f}%")
        print(f"Dice Score     : {results[f'{task}_dice'] * 100:.2f}%")
        print(f"Precision      : {results[f'{task}_precision'] * 100:.2f}%")
        print(f"Recall         : {results[f'{task}_recall'] * 100:.2f}%")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a specific test dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config (e.g. src/configs/phase2_building.yaml)")
    parser.add_argument("--test_data", type=str, default=None, help="Path to the directory containing new test images & masks")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If test_data is not provided via CLI, it will fall back to reading it from the config inside evaluate()
    evaluate(args.config, args.test_data, device)
