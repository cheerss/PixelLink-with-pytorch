import numpy as np
def ImgColorFormat(img, from_color="RGB", to_color="BGR"):
    if img.shape[0] == 3:
        order_format = "CHW"
    elif img.shape[2] == 3:
        order_format = "HWC"
    else:
        raise ValueError("unknown order format of shape " + str(img.shape))
    if (from_color == "RGB" and to_color == "BGR") or (from_color == "BGR" and to_color == "RGB"):
        if order_format == "HWC":
            return img[..., [2, 1, 0]]
        elif order_format == "CHW":
            return img[[2, 1, 0], ...]
    else:
        raise ValueError("unknown color format %s or %s" % (from_color, to_color))

def ImgOrderFormat(img, from_order="HWC", to_order="CHW"):
    if from_order == "HWC" and to_order == "CHW":
        return img.transpose(2, 1, 0)
    elif from_order == "CHW" and to_order == "HWC":
        return img.transpose(1, 2, 0)
    else:
        raise ValueError("unknown order format %s or %s" % (from_order, to_order))