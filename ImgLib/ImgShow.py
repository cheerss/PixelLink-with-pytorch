import cv2
import numpy as np

def DrawLabels(img, labels, data_format="HWC", color_format="RGB", contour_color=(0, 255, 0), width=1):
    """
    img: numpy array of image(HWC)

    return img(BGR)
    """
    labels = np.array(labels["coor"])
    labels = labels.reshape([-1, 4, 2])
    if color_format == "RGB":
        img = img[..., [2, 1, 0]]
    if data_format == "CHW":
        img = img.transpose(1, 2, 0)
    img = np.ascontiguousarray(img)
    cv2.drawContours(img, labels, contourIdx=-1, color=contour_color, thickness=width)
    return img