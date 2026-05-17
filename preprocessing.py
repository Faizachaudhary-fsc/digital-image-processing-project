#prepare image for segmentation 
import cv2

def preprocess_image(img):
 
    #Resize, convert to grayscale and remove noise
  
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, blur
