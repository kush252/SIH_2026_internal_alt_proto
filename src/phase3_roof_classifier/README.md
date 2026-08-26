# Phase 3: Roof Classification

This module provides the training and inference pipeline for Phase 3 of the SVAMITVA Aerial Mapping system. 
It uses a `ConvNeXt-Tiny` model to classify cropped roof images into one of four categories: `RCC`, `TILED`, `TIN`, `OTHER`.

## Dataset Constraints (Critical)
The model is trained on the RoofNet dataset (Kaggle xBD crops). 
Due to Google Maps licensing restrictions, 14,000 images are missing from the public dataset. Furthermore, the dataset **does not contain building polygons**, only pre-cropped rectangular images.
Therefore, the model is trained purely on **rectangular RGB crops** and *not* masked polygons. 

## Phase 2 → Phase 3 Inference Interface
During Phase 2 inference, your segmentation model will produce building polygons.
To pass these polygons into Phase 3, use the `Phase3Predictor` class in `scripts/predict.py`.

```python
from phase3_roof_classifier.scripts.predict import Phase3Predictor

predictor = Phase3Predictor(checkpoint_path="outputs/best_model.pth")

# Full orthophoto image + OpenCV contour polygons from Phase 2
results = predictor.predict_roofs(
    image_path="svamitva_uav_image.png",
    polygons=[poly1, poly2, poly3]
)

for res in results:
    print(f"Predicted: {res['roof_type']} with Confidence {res['confidence']}")
```

## Training the Model
1. Ensure the RoofNet dataset is available at `data/roofnet/`.
2. Generate the geographic split to prevent leakage (already done, saved to `datasets/roofnet_phase3_split.csv`).
3. Run the training script:
```bash
python -m phase3_roof_classifier.scripts.train
```

Outputs, including `best_model.pth` and `history.json`, will be saved to `phase3_roof_classifier/outputs/`.
