from tkinter import Tk, filedialog
import cv2

from preprocessing import preprocess_image
from segmentation import segment_foreground
from morphology import refine_mask
from background_replace import replace_background

# Hide tkinter main window
Tk().withdraw()

# Select foreground image
fg_path = filedialog.askopenfilename(
    title="Select Foreground Image",
    filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
)

if fg_path == "":
    print("No foreground image selected")
    exit()

# Select background image
bg_path = filedialog.askopenfilename(
    title="Select Background Image",
    filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
)

if bg_path == "":
    print("No background image selected")
    exit()

# Read images
fg_img = cv2.imread(fg_path)
bg_img = cv2.imread(bg_path)

# Preprocessing
original, blurred = preprocess_image(fg_img)

#segementation
mask = segment_foreground(original)


# Mask refinement
clean_mask = refine_mask(mask)

# Background replacement
result = replace_background(original, clean_mask, bg_img)

# Display results
cv2.imshow("Original Image", original)
cv2.imshow("Binary Mask", clean_mask)
cv2.imshow("Final Output", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
