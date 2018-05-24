import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import datasets
import config
import torch
import numpy as np
import cv2

save_path = os.path.abspath(os.path.dirname(__file__))

def test_pixel_mask(sample):
    # dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    # sample = dataset.__getitem__(588)
    image = sample['image'].data.numpy() * 255 # C * H * W
    image = image[(2, 1, 0), ...] # RGB to BGR
    image = np.transpose(image, (1, 2, 0)) # H * W * C
    shape = image.shape
    image = image.reshape([int(shape[0]/2), 2, int(shape[1]/2), 2, shape[2]])
    image = image.max(axis=(1, 3))
    cv2.imwrite(save_path + "/pixel_origin588.jpg", image)

    mask = sample['pixel_mask'] # H * W torch tensor
    mask = torch.unsqueeze(mask, 2).expand(-1, -1, 3).numpy()
    image[mask>0] = 0
    image = np.ascontiguousarray(image)
    cv2.imwrite(save_path + "/pixel_mask588.jpg", image)

def test_link_mask(sample):
    link_mask = sample['link_mask']
    image = sample['image'].data.numpy() * 255 # C * H * W
    image = image[(2, 1, 0), ...] # RGB to BGR
    image = np.transpose(image, (1, 2, 0)) # H * W * C
    shape = image.shape
    image = image.reshape([int(shape[0]/2), 2, int(shape[1]/2), 2, shape[2]])
    image = image.max(axis=(1, 3))

    for i in range(8):
        link = link_mask[i]
        link = 1 - link
        link = torch.unsqueeze(link, 2).expand(-1, -1, 3).numpy()
        image = image * link
        cv2.imwrite(save_path + "/link_mask588_%d.jpg" % i, image)


def test_pixel_weight(sample):
    # dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    # sample = dataset.__getitem__(588)
    image = sample['image'].data.numpy() * 255 # C * H * W
    image = image[(2, 1, 0), ...] # RGB to BGR
    image = np.transpose(image, (1, 2, 0)) # H * W * C
    shape = image.shape
    image = image.reshape([int(shape[0]/2), 2, int(shape[1]/2), 2, shape[2]])
    image = image.max(axis=(1, 3))
    cv2.imwrite(save_path + "/weight_origin588.jpg", image)

    weight_mask = sample['pixel_pos_weight']
    weight_mask = 1 - weight_mask
    weight_mask = torch.unsqueeze(weight_mask, 2).expand(-1, -1, 3).numpy()
    image = image * weight_mask
    cv2.imwrite(save_path + "/weight_mask588.jpg", image)

def main():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    sample = dataset.__getitem__(588)
    test_pixel_mask(sample)
    test_pixel_weight(sample)
    test_link_mask(sample)

if __name__ == '__main__':
    main()