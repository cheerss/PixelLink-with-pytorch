from PIL import Image
import numpy as np
import random
import math

def ReadImage(filename, output_format="numpy", order="HWC", color_format="RGB"):
    """
    filename: filepath+filename.jpg/png/...
    output_format: "Pillow" | "numpy"(default)
    order： "CHW" | "HWC"(default)
    color_format: "RGB"(default) | "BGR"

    return: Pillow image object | numpy array
    """
    with Image.open(filename) as img:
        image = np.array(img)
        if color_format == "RGB":
            pass
        elif color_format == "BGR":
            image = image[:, :, (2, 1, 0)]
        else:
            ValueError("Unknown color_format '{}'".format(color_format))

        if order == "CHW":
            image = image.transpose(2, 0, 1)
        elif order == "HWC":
            pass
        else:
            ValueError("Unknown order '{}'".format(order))

        if output_format == "numpy":
            return image
        elif output_format == "Pillow":
            img = img.load()
            return img
        else:
            ValueError("Unknown output_format '{}'".format(output_format))
    return

def _ParseImage(filename=None, data=None):
    """
    parse image according to the given type
    """
    if data is not None:
        img = Image.fromarray(np.uint8(data))
    elif filename is not None:
        img = ReadImage(filename, output_format="Pillow", order="HWC", color_format="RGB")
    else:
        raise ValueError("either filename or data should be given")
    return img

def _ReturnImage(img, return_type):
    """
    return image according to the required type
    """
    if return_type == "numpy":
        return np.array(img)
    elif return_type == "Pillow":
        return img
    else:
        raise ValueError("Unknown return type")

def RotateImage(filename=None, data=None, angle=None, return_type="numpy"):
    """
    either filename or data should be given
    filename: filepath+filename.jpg/png/...
    data: numpy array of data with type uint8
    angle: angle to rotate, None for random(default)

    return: [numpy array, rotate_angle]
    """
    img = _ParseImage(filename, data)

    if angle is None:
        rotate_rand = random.randint(0, 3)
        angle = rotate_rand * 90
    else:
        angle = angle // 90 * 90
    img = img.rotate(angle, expand=True)
    return _ReturnImage(img, return_type), angle

def RotateImageWithLabel(labels, filename=None, data=None, angle=None, return_type="numpy"):
    """
    labels: a dict with item(key="coor", val=list or coors), 4 corners for 8 vals, 2 corners for 4 vals

    return: labels, numpy array, rotate_angle
    """
    img = _ParseImage(filename, data)
    origin_w, origin_h = img.size
    img, angle = RotateImage(filename=filename, data=data, angle=angle, return_type=return_type)
    new_label = labels["coor"]
    new_label = np.array(new_label)
    new_label = new_label.reshape([-1, 4, 2])
    for i in range(angle // 90):
        new_label[:, :, 0] = origin_w - 1 - new_label[:, :, 0]
        new_label[:, :, [0, 1]] = new_label[:, :, [1, 0]]
        origin_h, origin_w = origin_w, origin_h
    new_label = new_label.reshape([-1, 8]).astype(int)
    labels["coor"] = new_label.tolist()
    return labels, img, angle


def CropImage(filename=None, data=None, start=None, end=None, scale=None, ratio=None, return_type="numpy"):
    """
    either filename or data should be given, either start/end or scale should be given
    start: [w_index, h_index]
    end: [w_index, h_index], should > start and < image shape
    scale: 0.1 ~ 0.9
    ratio: new w/h, None for keep original ratio(default)

    return: img, start + end
    """
    img = _ParseImage(filename, data)
    if (start is not None) and (end is not None):
        img = img.crop(start[0], start[1], end[0], end[1])
        return img, [start[0], start[1], end[0], end[1]]
    w, h = img.size
    area = w * h
    if ratio is None:
        ratio = 0.5 + random.random() * 1.5
    if scale is None:
        raise ValueError("either start/end or scale should be given")
    new_area = area * scale
    for attempt in range(10):  
        new_h = int(round(math.sqrt(new_area / ratio)))
        new_w = int(round(math.sqrt(new_area * ratio)))
        if new_h < h and new_w < w:
            new_h_start = random.randint(0, h - new_h)
            new_w_start = random.randint(0, w - new_w)
            break
    else:
        new_w = min(h, w)
        new_h = new_w
        new_h_start = (h - new_h) // 2
        new_w_start = (w - new_w) // 2
    start = [new_w_start, new_h_start]
    end = [new_w_start + new_w, new_h_start + new_h]
    img = img.crop((new_w_start, new_h_start, new_w_start + new_w, new_h_start + new_h))
    return _ReturnImage(img, return_type), start + end

def CropImageWithLabel(labels, filename=None, data=None, start=None, end=None, scale=None, ratio=None, return_type="numpy"):
    """
    return: labels, img, img_range
    """
    img, img_range = CropImage(filename=filename, data=data, start=start, end=end, \
        scale=scale, ratio=ratio, return_type=return_type)

    new_h = img_range[3] - img_range[1]
    new_w = img_range[2] - img_range[0]
    new_label = labels["coor"]
    new_label = np.array(new_label)
    new_label = new_label.reshape([-1, 4, 2])
    new_label[:, :, 1] -= img_range[1]
    new_label[:, :, 0] -= img_range[0]
    new_label[new_label < 0] = 0
    new_label[:, :, 1][new_label[:, :, 1] >= new_h] = new_h - 1
    new_label[:, :, 0][new_label[:, :, 0] >= new_w] = new_w - 1
    new_label = new_label.reshape([-1, 8]).astype(int)
    labels["coor"] = new_label.tolist()
    return labels, img, img_range

def ResizeImage(size, filename=None, data=None, return_type="numpy"):
    """
    either filename or data should be given
    size: new image's size, (w, h)

    return:
    """
    img = _ParseImage(filename, data)
    img = img.resize(size, resample=Image.BILINEAR)
    return _ReturnImage(img, return_type), size

def ResizeImageWithLabel(labels, size, filename=None, data=None, return_type="numpy"):
    """
    
    """
    origin_img = _ParseImage(filename, data)
    w, h = origin_img.size
    img, size = ResizeImage(size, filename=filename, data=data, return_type=return_type)
    new_label = labels["coor"]
    new_label = np.array(new_label)
    new_label = new_label.reshape([-1, 4, 2])
    new_label[:, :, 1] = new_label[:, :, 1] * size[1] / h
    new_label[:, :, 0] = new_label[:, :, 0] * size[0] / w
    new_label = new_label.reshape([-1, 8]).astype(int)
    labels["coor"] = new_label.tolist()
    return labels, img, size

def ZeroMeanImage(img, r_mean, g_mean, b_mean):
    """
    img: numpy array of H*W*C
    """
    img = img.astype(np.float)
    img[..., 0] -= r_mean
    img[..., 1] -= g_mean
    img[..., 2] -= b_mean
    return img

def UnzeroMeanImage(img, r_mean, g_mean, b_mean):
    """
    img: numpy array of H*W*C
    """
    img = img.astype(np.float)
    img[..., 0] += r_mean
    img[..., 1] += g_mean
    img[..., 2] += b_mean
    return img






