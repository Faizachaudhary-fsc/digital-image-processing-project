import cv2
import numpy as np

def segment_foreground(img):
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    
    h, w = img.shape[:2]

    rect = (
    int(w * 0.15),   # x (more left space)
    int(h * 0.05),   # y
    int(w * 0.7),    # width (wider)
    int(h * 0.9)     # height
)

    cv2.grabCut(
        img,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    final_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    return final_mask
