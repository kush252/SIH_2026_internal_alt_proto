# SIH 2026 Remote Sensing Segmentation: Architecture Migration

## Overview

This repository contains the ongoing production-level architecture migration for the SVAMITVA remote-sensing segmentation system. 

The objective is to create and evaluate a significantly simpler CNN-based alternative to the previous Transformer-based architecture, focusing on strong building/road/water footprint extraction while lowering computational and memory requirements.

### Architecture Comparison

| Component | Old Architecture | New Architecture (This Branch) |
| :--- | :--- | :--- |
| **Encoder** | Swin Transformer (Swin-T) | ResNet-50 |
| **Phase 1 SSL** | SimMIM | SimSiam |
| **Decoder (Phase 2)** | Mask2Former | DeepLabV3+ |

## Migration Status

> [!WARNING]
> **Phase 1 is fully migrated.** The encoder pretraining now uses `ResNet50 + SimSiam`.
> 
> **Phase 2 training is intentionally pending.** Do NOT run the Phase 2 training pipeline expecting it to use the new architecture yet. The Phase 2 code structurally depends on `models/old/` to prevent breakage until the DeepLabV3+ migration is complete.
>
> **Phase 3 roof classifier remains ConvNeXt-based** and is unaffected by this migration.

### Why ResNet50 + SimSiam?
- **ResNet50:** Provides multi-scale convolutional features (C2, C3, C4, C5) that are highly effective for fine boundary preservation in high-resolution UAV imagery. It is computationally lighter and easier to deploy than Swin.
- **SimSiam:** Naturally compatible with CNN encoders, does not require large negative-sample queues (unlike MoCo/SimCLR), and provides a clean encoder-pretraining → downstream-transfer workflow suitable for limited GPU resources.
- **DeepLabV3+ (Future):** ASPP provides multi-scale context, utilizing low-level spatial information crucial for buildings and roads without the complex instance-segmentation overhead of Mask2Former.

## Running Phase 1 (SimSiam)

Phase 1 pretrains the ResNet50 encoder using self-supervised learning on augmented two-view image pairs.

### Command
```bash
python src/train_phase1.py --config src/configs/simsiam_resnet50.yaml
```

### Expected Checkpoint Outputs
After training, two checkpoints will be available (via `extract_encoder.py`):
- `best.pt`: Full SimSiam training state (encoder + projector + predictor + optimizer).
- `resnet50_simsiam_encoder.pt`: Transferable ResNet50 encoder weights ONLY, ready for Phase 2 initialization.

## Testing & Smoke Test

A comprehensive test suite verifies the migration components, ensuring shapes, gradient flows, AMP compatibility, and learning capabilities.

### Command
```bash
python src/test_migration.py
```

## Legacy Code

The legacy `Swin` and `Mask2Former` code has been cleanly isolated in `src/models/old/`. The datasets and configuration systems remain compatible with the old architecture for robust comparison.
