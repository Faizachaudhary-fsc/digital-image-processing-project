import cv2
import numpy as np

def refine_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
