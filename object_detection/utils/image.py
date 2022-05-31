import cv2
import numpy as np

def load_image(img_path: str) -> np.ndarray:
    try:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception:
        raise (f"Could not load image, path - {img_path}")
