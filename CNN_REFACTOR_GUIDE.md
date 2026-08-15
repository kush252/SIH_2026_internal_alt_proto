# CNN Refactor Guide for AI Agent

## Objective
We are migrating from a complex Transformer-based (Mask2Former + Swin) architecture to a lightweight, fast CNN-based architecture (e.g., U-Net). The goal is to retain the **extremely robust data pipeline and training loop** while completely replacing the model, loss, and inference logic. 

**INSTRUCTIONS FOR AGENT:** Follow this guide strictly. Do not reinvent the wheel. We are keeping the robust foundation and only surgically swapping the model engine.

## 1. What NOT to Change (The Robust Foundation)
Do **NOT** modify the core logic of these files. They are highly optimized and architecture-agnostic:
- **`datasets/generic_binary_dataset.py`**: This perfectly handles SVAMITVA (folders), DeepGlobe (mixed files), and BONAI (on-the-fly JSON polygon rendering). It provides a universal `targets` dictionary. Keep it exactly as is!
- **`training/metrics.py`**: The `MultitaskMetrics` class is fully robust. It automatically detects raw CNN logits (values outside [0,1]) and applies a Sigmoid activation automatically. Keep it as is!
- **`train_phase2.py` (Main Entrypoint)**: The CLI arguments (`--task`, `--resume`, `--config`), dynamic YAML overriding, and dataset instantiation are perfect. Do not change them.

## 2. What to COMPLETELY Replace (The Architecture)
These files are tied to Mask2Former and need to be completely rewritten for a CNN:
- **`models/mask2former.py`, `models/swin_encoder.py`, `models/phase1_encoder_loader.py`**: DELETE these entirely.
- **`models/task_heads.py`**: Replace `Phase2MultiTaskModel` with a standard CNN (like `UNet`). The new `forward` pass should simply return `raw_logits` of shape `[B, 1, H, W]`. Remove the complex `semantic_inference` dot-product method completely.
- **`losses/multitask_loss.py`**: DELETE the `SetCriterion` and Hungarian matching. Replace it with a simple, standard combination of `BCEWithLogitsLoss` + `DiceLoss`.
- **`configs/phase2_building.yaml`**: Remove the `MODEL.encoder` and `MODEL.decoder` sections. Add `MODEL.backbone` (e.g., resnet34) and `MODEL.architecture` (e.g., Unet).

## 3. What to MODIFY (The Glue)
These files need surgical updates to connect the new CNN to the old robust pipeline:
- **`training/trainer_phase2.py`**:
  - **Optimizer:** Remove the differential learning rates (`encoder_params`, `decoder_params`, `head_params`) in `__init__`. Use a standard `optimizer = torch.optim.AdamW(model.parameters(), ...)`
  - **Loss Call:** In `train_epoch` and `validate`, the new CNN will directly output logits: `preds = self.model(images)`. Change the loss calculation to simply `loss = self.criterion(preds, targets)`.
  - **Metrics Call:** In `validate`, remove the line `semantic_preds = self.model.semantic_inference(...)`. Instead, just pass the raw CNN predictions directly to the metrics: `self.metrics.update(preds, targets)`. The metrics script will handle the sigmoid activation automatically!

## 4. Execution Steps for the Agent
1. Copy the entire `src` folder to `src_cnn`.
2. Delete the complex Transformer model files inside `src_cnn/models`.
3. Create `src_cnn/models/cnn_model.py` (implementing a standard U-Net).
4. Rewrite `src_cnn/losses/cnn_loss.py` (implementing BCE + Dice).
5. Surgically update `src_cnn/training/trainer_phase2.py` to remove Mask2Former logic as described above.
6. Run a dry run to verify the pipeline trains flawlessly!
