#combine foreground + new background
import cv2

def replace_background(fg_img, mask, bg_img):
    """
    Replace background using mask
    """
    bg_img = cv2.resize(bg_img, (fg_img.shape[1], fg_img.shape[0]))

    mask_inv = cv2.bitwise_not(mask)

    fg_part = cv2.bitwise_and(fg_img, fg_img, mask=mask)
    bg_part = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)

    final_img = cv2.add(fg_part, bg_part)
    return final_img
