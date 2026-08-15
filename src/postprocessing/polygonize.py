import cv2
import numpy as np

def mask_to_polygons(binary_mask, min_area=10.0):
    """
    Converts a binary mask (H, W) into a list of polygons.
    binary_mask: numpy array (H, W) with values 0 or 1.
    """
    # Ensure uint8
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            # Approximate the contour to reduce points (cadastral style)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Squeeze and convert to list of [x, y] coordinates
            poly = approx.squeeze(axis=1).tolist()
            
            if len(poly) >= 3:
                polygons.append(poly)
                
    return polygons
