import cv2
import numpy as np

def find_contours(mask, method = None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP, 
                                   method = method)
    except:
        _, contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP, 
                                  method = method)
    return contours