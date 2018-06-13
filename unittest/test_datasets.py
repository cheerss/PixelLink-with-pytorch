import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import datasets
import config
import torch
import numpy as np
import cv2
import argparse
import ImgLib.ImgTransform as ImgTransform

save_path = os.path.abspath(os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("index", help="index of the image", type=int)
args = parser.parse_args()

def test_pixel_mask(sample, index, version="2s"):
    # dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    # sample = dataset.__getitem__(588)
    factor = 2 if version=="2s" else 4
    image = sample['image'].data.numpy() # C * H * W
    image = np.transpose(image, (1, 2, 0)) # H * W * C
    image = ImgTransform.UnzeroMeanImage(image, config.r_mean, config.g_mean, config.b_mean)
    image = image[..., (2, 1, 0)] # RGB to BGR
    shape = image.shape
    image = image.reshape([int(shape[0]/factor), factor, int(shape[1]/factor), factor, shape[2]])
    image = image.max(axis=(1, 3))
    cv2.imwrite(save_path + "/pixel_origin%d.jpg" % index, image)

    mask = sample['pixel_mask'] # H * W torch tensor
    mask = torch.unsqueeze(mask, 2).expand(-1, -1, 3).numpy()
    image[mask>0] = 0
    image = np.ascontiguousarray(image)
    cv2.imwrite(save_path + "/pixel_mask%d.jpg" % index, image)

def test_link_mask(sample, index, version="2s"):
    factor = 2 if version=="2s" else 4
    link_mask = sample['link_mask']
    image = sample['image'].data.numpy() # C * H * W
    image = np.transpose(image, (1, 2, 0)) # H * W * C
    image = ImgTransform.UnzeroMeanImage(image, config.r_mean, config.g_mean, config.b_mean)
    image = image[..., (2, 1, 0)] # RGB to BGR
    shape = image.shape
    image = image.reshape([int(shape[0]/factor), factor, int(shape[1]/factor), factor, shape[2]])
    image = image.max(axis=(1, 3))

    for i in range(8):
        link = link_mask[i]
        link = 1 - link
        link = torch.unsqueeze(link, 2).expand(-1, -1, 3).numpy()
        image = image * link
        cv2.imwrite(save_path + "/link_mask%d_%d.jpg" % (index, i), image)


def test_pixel_weight(sample, index, version="2s"):
    factor = 2 if version=="2s" else 4
    # dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    # sample = dataset.__getitem__(588)
    image = sample['image'].data.numpy() # C * H * W
    image = np.transpose(image, (1, 2, 0)) # H * W * C
    image = ImgTransform.UnzeroMeanImage(image, config.r_mean, config.g_mean, config.b_mean)
    image = image[..., (2, 1, 0)] # RGB to BGR
    shape = image.shape
    image = image.reshape([int(shape[0]/factor), factor, int(shape[1]/factor), factor, shape[2]])
    image = image.max(axis=(1, 3))
    cv2.imwrite(save_path + "/weight_origin%d.jpg" % index, image)

    weight_mask = sample['pixel_pos_weight']
    weight_mask = 1 - weight_mask
    weight_mask = torch.unsqueeze(weight_mask, 2).expand(-1, -1, 3).numpy()
    image = image * weight_mask
    cv2.imwrite(save_path + "/weight_mask%d.jpg" % index, image)

def test_label(sample, index, version="2s"):
    factor = 2 if version=="2s" else 4
    image = sample['image'].data.numpy() # C * H * W
    image = np.transpose(image, (1, 2, 0)) # H * W * C
    image = ImgTransform.UnzeroMeanImage(image, config.r_mean, config.g_mean, config.b_mean)
    image = image[..., (2, 1, 0)] # RGB to BGR
    image = np.ascontiguousarray(image)

    for i, label in enumerate(sample['label']["coor"]):
        label = np.array(label)
        label = label.reshape(1, 4, 2)
        if sample['label']['ignore'][i]:
            cv2.drawContours(image, label, contourIdx=-1, color=(0, 0, 255), thickness=1)
        else:
            cv2.drawContours(image, label, contourIdx=-1, color=(0, 255, 0), thickness=1)
    cv2.imwrite(save_path + "/label%d.jpg" % index, image)

def main():
    index = args.index
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir, train=False)
    sample = dataset.__getitem__(index)
    test_pixel_mask(sample, index, version=config.version)
    test_pixel_weight(sample, index, version=config.version)
    test_link_mask(sample, index, version=config.version)
    test_label(sample, index, version=config.version)

if __name__ == '__main__':
    main()