import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from phase3_roof_classifier.models.classifier import RoofClassifier

class Phase3Predictor:
    def __init__(self, checkpoint_path, model_name='convnext_tiny', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = RoofClassifier(model_name=model_name, num_classes=4, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.idx_to_class = {0: 'RCC', 1: 'TILED', 2: 'TIN', 3: 'OTHER'}
        
    def predict_roofs(self, image_path, polygons):
        """
        Phase 2 to Phase 3 Interface.
        Args:
            image_path (str): Path to the original full UAV/Satellite image.
            polygons (list of np.ndarray): List of OpenCV contour polygons of buildings.
        Returns:
            list of dicts containing the original polygon and its predicted roof type.
        """
        try:
            full_img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return []
            
        full_img_np = np.array(full_img)
        results = []
        
        for poly in polygons:
            # 1. Calculate bounding box for the polygon
            x, y, w, h = cv2.boundingRect(poly)
            
            # 2. Add a 10% margin
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(full_img_np.shape[1], x + w + margin_x)
            y2 = min(full_img_np.shape[0], y + h + margin_y)
            
            # 3. Crop the rectangular region (we no longer mask out the background 
            # because the Kaggle model was only trained on raw rectangular crops)
            crop_np = full_img_np[y1:y2, x1:x2]
            
            if crop_np.size == 0:
                continue
                
            crop_pil = Image.fromarray(crop_np)
            tensor_img = self.transforms(crop_pil).unsqueeze(0).to(self.device)
            
            # 4. Predict
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    outputs = self.model(tensor_img)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
            pred_class = self.idx_to_class[pred_idx.item()]
            
            results.append({
                'polygon': poly,
                'bbox': (x1, y1, x2, y2),
                'roof_type': pred_class,
                'confidence': conf.item()
            })
            
        return results

if __name__ == "__main__":
    print("Phase 3 Predictor module loaded successfully.")
