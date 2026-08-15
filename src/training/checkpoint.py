import torch
import os

def save_checkpoint(model, optimizer, scheduler, epoch, step, config, output_dir, is_best=False):
    """Save training checkpoint and a separate encoder-only checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle DataParallel
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    state = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config
    }
    
    latest_path = os.path.join(output_dir, 'latest.pt')
    torch.save(state, latest_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'best.pt')
        torch.save(state, best_path)
        
        # Save encoder-only weights for downstream tasks
        # Encoder is inside SimMIM, and if DataParallel is used, it's model.module.encoder
        base_model = model.module if hasattr(model, 'module') else model
        encoder_path = os.path.join(output_dir, 'swin_t_simmim_encoder.pt')
        torch.save(base_model.encoder.state_dict(), encoder_path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle DataParallel loading
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)
