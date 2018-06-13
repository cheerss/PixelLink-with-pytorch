from torch.utils.data import Dataset
from PIL import Image
import config
import torch
import codecs
import random
import math
import copy
import time
import cv2
import os
import numpy as np
from torchvision import transforms
import ImgLib.ImgTransform as ImgTransform

class ICDAR15Dataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        # self.all_images = self.read_datasets(images_dir, config.all_trains)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.all_labels = self.read_labels(labels_dir, config.all_trains)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        if isinstance(index, int):
            return {'image': self.read_image(self.images_dir, index), 'label': all_labels[index]}

    def read_image(self, dir, index):
        index += 1
        filename = os.path.join(dir, "img_" + str(index) + ".jpg")
        image = ImgTransform.ReadImage(filename)
        return image

    def read_datasets(self, dir, num):
        res = []
        for i in range(1, num+1):
            image = Image.open(dir+ "img_" + str(i) + ".jpg")
            res.append(image)
            if i % 100 == 0:
                print(i)
        # print(res[0].shape)
        return res

    def read_labels(self, dir, num):
        res = [[] for i in range(num)]
        for i in range(1, num+1):
            # utf-8_sig for bom_utf-8
            # print("read %d" % i)

            with codecs.open(dir + "gt_img_" + str(i) + ".txt", encoding="utf-8_sig") as file:
                data = file.readlines()
                tmp = {}
                tmp["coor"] = []
                tmp["content"] = []
                tmp["ignore"] = []
                tmp["area"] = []
                for line in data:
                    content = line.split(",")
                    coor = [int(n) for n in content[:8]]
                    tmp["coor"].append(coor)
                    content[8] = content[8].strip("\r\n")
                    tmp["content"].append(content[8])
                    if content[8] == "###":
                        tmp["ignore"].append(True)
                    else:
                        tmp["ignore"].append(False)
                    coor = np.array(coor).reshape([4,2])
                    tmp["area"].append(cv2.contourArea(coor))
                res[i-1] = tmp
        return res

class PixelLinkIC15Dataset(ICDAR15Dataset):
    def __init__(self, images_dir, labels_dir, train=True):
        super(PixelLinkIC15Dataset, self).__init__(images_dir, labels_dir)
        self.train = train
        # self.all_images = torch.Tensor(self.all_images)

    def __getitem__(self, index):
        # print(index, end=" ")
        if self.train:
            image, label = self.train_data_transform(index)
        else:
            image, label = self.test_data_transform(index)
        image = torch.Tensor(image)

        pixel_mask, neg_pixel_mask, pixel_pos_weight, link_mask = \
            PixelLinkIC15Dataset.label_to_mask_and_pixel_pos_weight(label["coor"], list(image.shape[1:]), version=config.version)
        return {'image': image, 'pixel_mask': pixel_mask, 'neg_pixel_mask': neg_pixel_mask, 'label': label,
                'pixel_pos_weight': pixel_pos_weight, 'link_mask': link_mask}

    def test_data_transform(self, index):
        img = self.read_image(self.images_dir, index)
        labels = self.all_labels[index]
        labels, img, size = ImgTransform.ResizeImageWithLabel(labels, (512, 512), data=img)
        img = ImgTransform.ZeroMeanImage(img, config.r_mean, config.g_mean, config.b_mean)
        img = img.transpose(2, 0, 1)
        return img, labels

    def train_data_transform(self, index):
        img = self.read_image(self.images_dir, index)
        labels = self.all_labels[index]

        rotate_rand = random.random() if config.use_rotate else 0
        crop_rand = random.random() if config.use_crop else 0
        # rotate
        if rotate_rand > 0.5:
            labels, img, angle = ImgTransform.RotateImageWithLabel(labels, data=img)
        # crop
        if crop_rand > 0.5:
            scale = 0.1 + random.random() * 0.9
            labels, img, img_range = ImgTransform.CropImageWithLabel(labels, data=img, scale=scale)
            labels = PixelLinkIC15Dataset.filter_labels(labels, method="rai")
        # resize
        labels, img, size = ImgTransform.ResizeImageWithLabel(labels, (512, 512), data=img)
        # filter unsatifactory labels
        labels = PixelLinkIC15Dataset.filter_labels(labels, method="msi")
        # zero mean
        img = ImgTransform.ZeroMeanImage(img, config.r_mean, config.g_mean, config.b_mean)
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        return img, labels

    @staticmethod
    def filter_labels(labels, method):
        """
        method: "msi" for min area ignore, "rai" for remain area ignore
        """
        def distance(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        def min_side_ignore(label):
            label = np.array(label).reshape(4, 2)
            dists = []
            for i in range(4):
                dists.append(distance(label[i], label[(i+1)%4]))
            if min(dists) < 10:
                return True # ignore it
            else:
                return False

        def remain_area_ignore(label, origin_area):
            label = np.array(label).reshape(4, 2)
            area = cv2.contourArea(label)
            if area / origin_area < 0.2:
                return True
            else:
                return False
        if method == "msi":
            ignore = list(map(min_side_ignore, labels["coor"]))
        elif method == "rai":
            ignore = list(map(remain_area_ignore, labels["coor"], labels["area"]))
        else:
            ignore = [False] * 8
        labels["ignore"] = list(map(lambda a, b: a or b, labels["ignore"], ignore))
        return labels

    @staticmethod
    def label_to_mask_and_pixel_pos_weight(label, img_size, version="2s", neighbors=8):
        """
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        """
        factor = 2 if version == "2s" else 4

        label = np.array(label)
        label = label.reshape([-1, 1, 4, 2])
        pixel_mask_size = [int(i / factor) for i in img_size]
        link_mask_size = [neighbors, ] + pixel_mask_size

        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_weight = np.zeros(pixel_mask_size, dtype=np.float)
        link_mask = np.zeros(link_mask_size, dtype=np.uint8)
        # if label.shape[0] == 0:
            # return torch.LongTensor(pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)
        label = (label / factor).astype(int) # label's coordinate value should be divided

        # cv2.drawContours(pixel_mask, label, -1, 1, thickness=-1)
        real_box_num = 0
        # area_per_box = []
        for i in range(label.shape[0]):
            pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
            cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1)
            pixel_mask += pixel_mask_tmp
        neg_pixel_mask = (pixel_mask == 0).astype(np.uint8)
        pixel_mask[pixel_mask != 1] = 0
        # assert not (pixel_mask>1).any()
        pixel_mask_area = np.count_nonzero(pixel_mask) # total area

        for i in range(label.shape[0]):
            pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
            cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1)
            pixel_mask_tmp *= pixel_mask
            if np.count_nonzero(pixel_mask_tmp) > 0:
                real_box_num += 1
        if real_box_num == 0:
            return torch.LongTensor(pixel_mask), torch.LongTensor(neg_pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)
        avg_weight_per_box = pixel_mask_area / real_box_num

        for i in range(label.shape[0]): # num of box
            pixel_weight_tmp = np.zeros(pixel_mask_size, dtype=np.float)
            cv2.drawContours(pixel_weight_tmp, [label[i]], -1, avg_weight_per_box, thickness=-1)
            pixel_weight_tmp *= pixel_mask
            area = np.count_nonzero(pixel_weight_tmp) # area per box
            if area <= 0:
                  # print("area label: " + str(label[i]))
                  # print("area:" + str(area))
                  continue
            pixel_weight_tmp /= area
            # print(pixel_weight_tmp[pixel_weight_tmp>0])
            pixel_weight += pixel_weight_tmp

            # link mask
            weight_tmp_nonzero = pixel_weight_tmp.nonzero()
            # pixel_weight_nonzero = pixel_weight.nonzero()
            link_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
            # for j in range(neighbors): # neighbors directions
            link_mask_tmp[weight_tmp_nonzero] = 1
            link_mask_shift = np.zeros(link_mask_size, dtype=np.uint8)
            w_index = weight_tmp_nonzero[1]
            h_index = weight_tmp_nonzero[0]
            w_index1 = np.clip(w_index + 1, a_min=None, a_max=link_mask_size[1] - 1)
            w_index_1 = np.clip(w_index - 1, a_min=0, a_max=None)
            h_index1 = np.clip(h_index + 1, a_min=None, a_max=link_mask_size[2] - 1)
            h_index_1 = np.clip(h_index - 1, a_min=0, a_max=None)
            link_mask_shift[0][h_index1, w_index1] = 1
            link_mask_shift[1][h_index1, w_index] = 1
            link_mask_shift[2][h_index1, w_index_1] = 1
            link_mask_shift[3][h_index, w_index_1] = 1
            link_mask_shift[4][h_index_1, w_index_1] = 1
            link_mask_shift[5][h_index_1, w_index] = 1
            link_mask_shift[6][h_index_1, w_index1] = 1
            link_mask_shift[7][h_index, w_index1] = 1

            for j in range(neighbors):
                # +0 to convert bool array to int array
                link_mask[j] += np.logical_and(link_mask_tmp, link_mask_shift[j]).astype(np.uint8)
        return [torch.LongTensor(pixel_mask), torch.LongTensor(neg_pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)]

if __name__ == '__main__':
    start = time.time()
    dataset = PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    end = time.time()
    print("time to read datasets: " + str(end - start)) # about 0.12s

    start = time.time()
    sample = dataset.__getitem__(588)
    end = time.time()
    print("time to get 1000 items: " + str(end - start)) # about 34s

    # pixel_mask = sample['pixel_pos_weight']
    # link_mask = sample['link_mask']
    image = sample['image'].data.numpy() * 255
    image = np.transpose(image, (1, 2, 0))
    image = np.ascontiguousarray(image)
    # shape = image.shape
    # image = image.reshape([int(shape[0]/2), 2, int(shape[1]/2), 2, shape[2]])
    # image = image.max(axis=(1, 3))
    # cv2.imwrite("trans0.jpg", image)
    # pixel_mask = pixel_mask.unsqueeze(2).expand(-1, -1, 3)
    # pixel_mask = pixel_mask.numpy()
    # import IPython 
    # IPython.embed()
    # link_mask = link_mask.unsqueeze(3).expand(-1, -1, -1, 3)
    # link_mask = link_mask.numpy()
    # image = image * pixel_mask
    label = sample['label'].reshape([-1, 4, 2])
    cv2.drawContours(image, label, -1, (255, 255, 0))
    cv2.imwrite("trans1.jpg", image)