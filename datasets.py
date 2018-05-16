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
import numpy as np
from torchvision import transforms

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
        image = Image.open(dir + "img_" + str(index) + ".jpg")
        image.load()
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
            with codecs.open(dir + "gt_img_" + str(i) + ".txt", encoding="utf-8_sig") as file:
                data = file.readlines()
                tmp = []
                for line in data:
                    tmp.append([int(n) for n in line.split(",")[:8]])
                res[i-1] = tmp
        return res

class PixelLinkIC15Dataset(ICDAR15Dataset):
    def __init__(self, images_dir, labels_dir):
        super(PixelLinkIC15Dataset, self).__init__(images_dir, labels_dir)
        # self.all_images = torch.Tensor(self.all_images)

    def __getitem__(self, index):
        image, label = self.data_transform(index)
        image = transforms.ToTensor()(image)
        pixel_mask, pixel_pos_weight, link_mask = \
            PixelLinkIC15Dataset.label_to_mask_and_pixel_pos_weight(label, list(image.shape[1:]))
        return {'image': image, 'pixel_mask': pixel_mask,
                'pixel_pos_weight': pixel_pos_weight, 'link_mask': link_mask}

    def data_transform(self, index):
        while True:
            # image = self.all_images[index]
            image = self.read_image(self.images_dir, index)
            label = self.all_labels[index]
            rotate_rand = random.randint(0, 3)
            origin_h = image.size[1]
            origin_w = image.size[0]
            # rotate
            image = image.rotate(90 * rotate_rand, expand=True) # counter clockwise
            # image.save("rotate.jpg", "JPEG")
            aspect_ratio_rand = 0.5 + random.random() * 1.5
            # print("rotate " + str(90 * rotate_rand) + " degrees")

            # print(image.size)
            h = image.size[1]
            w = image.size[0]
            # print("original size(h, w): (" + str(h) + " " + str(w) + ")")
            area = h * w
            for attempt in range(10):
                scale_rand = 0.1 + random.random() * 0.9
                new_area = area * scale_rand
                new_h = int(round(math.sqrt(new_area / aspect_ratio_rand)))
                new_w = int(round(math.sqrt(new_area * aspect_ratio_rand)))
                if new_h < h and new_w < w:
                    new_h_start = random.randint(0, h - new_h)
                    new_w_start = random.randint(0, w - new_w)
                    break
            else:
                new_w = min(h, w)
                new_h = new_w
                new_h_start = (h - new_h) // 2
                new_w_start = (w - new_w) // 2
            # print("size after crop should be (h, w): (" + str(new_h) + " " + str(new_w) + ")")
            # print("start after crop(h, w): (" + str(new_h_start) + " " + str(new_w_start) + ")")

            # crop
            image = image.crop((new_w_start, new_h_start, new_w_start + new_w, new_h_start + new_h))
            # print("size after crop(h, w): (" + str(image.size[1]) + " " + str(image.size[0]) + ")")
            # image.save("crop.jpg", "JPEG")
            # resize
            image = image.resize((512, 512), resample=Image.BILINEAR)
            # image.save("resize.jpg", "JPEG")
            # print("size after resize(h, w): (" + str(image.size[1]) + " " + str(image.size[0]) + ")")

            new_label = copy.deepcopy(label)
            new_label = np.array(new_label)
            new_label = new_label.reshape([-1, 4, 2])
            # ground truth file: x along width, y along height, ours: x along height, y along width
            # new_label[:, :, [0, 1]] = new_label[:, :, [1, 0]]
            # print("origin new label: " + str(new_label))
            box_areas = []
            for i in range(new_label.shape[0]):
                box_areas.append(cv2.contourArea(new_label[i]))

            # label rotate counter clockwise
            for i in range(rotate_rand):
                new_label[:, :, 0] = origin_w - 1 - new_label[:, :, 0]
                new_label[:, :, [0, 1]] = new_label[:, :, [1, 0]]
                origin_h, origin_w = origin_w, origin_h

            # label crop
            new_label[:, :, 1] -= new_h_start
            new_label[:, :, 0] -= new_w_start
            new_label[new_label < 0] = 0
            new_label[:, :, 1][new_label[:, :, 1] >= new_h] = new_h - 1
            new_label[:, :, 0][new_label[:, :, 0] >= new_w] = new_w - 1

            # label resize
            new_label[:, :, 1] = new_label[:, :, 1] * 512 / new_h
            new_label[:, :, 0] = new_label[:, :, 0] * 512 / new_w
            new_label = new_label.astype(int)

            # delete the boxes which unsatisfy the conditions
            delete_index = []
            for i in range(new_label.shape[0]):
                box = new_label[i]
                # min_side = min(box[3] - box[1], box[6] - box[0])

                # if min_side < 10:
                #     delete_index.append(i)
                #     continue
                new_box_area = cv2.contourArea(box)
                if new_box_area / box_areas[i] < 0.2:
                    delete_index.append(i)
            new_label = np.delete(new_label, delete_index, 0)
            if new_label.shape[0] > 0:
                break
        assert (new_label >= 0).all()
        return image, new_label

    @staticmethod
    def label_to_mask_and_pixel_pos_weight(label, img_size, version="2s", neighbors=8):
        """
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        """
        factor = 2
        if version == "4s":
            factor = 4

        # label = np.array(label)
        label.reshape([-1, 1, 4, 2])
        pixel_mask_size = [int(i / factor) for i in img_size]
        link_mask_size = [neighbors, ] + pixel_mask_size

        pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
        pixel_weight = np.zeros(pixel_mask_size, dtype=np.float)
        link_mask = np.zeros(link_mask_size, dtype=np.uint8)
        # if label.shape[0] == 0:
            # return torch.LongTensor(pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)
        label = (label / factor).astype(int) # label's coordinate value should be divided

        cv2.drawContours(pixel_mask, label, -1, 1, thickness=-1)
        pixel_mask_area = np.count_nonzero(pixel_mask) # total area
        avg_weight_per_box = pixel_mask_area / label.shape[0]

        for i in range(label.shape[0]): # num of box
            pixel_weight_tmp = np.zeros(pixel_mask_size, dtype=np.float)
            cv2.drawContours(pixel_weight_tmp, [label[i]], -1, avg_weight_per_box, thickness=-1)
            weight_tmp_nonzero = pixel_weight_tmp.nonzero()
            weight_nonzero = pixel_weight.nonzero()
            # pixel_weight[weight_tmp_nonzero] = 0 # when overlapping, only field without overlapping counts
            # pixel_weight_tmp[weight_nonzero] = 0
            area = np.count_nonzero(pixel_weight_tmp) # area per box
            if area <= 0:
                  print("area label: " + str(label[i]))
                  print("area:" + str(area))
            pixel_weight_tmp /= area
            pixel_weight += pixel_weight_tmp

            # link mask
            weight_tmp_nonzero = pixel_weight_tmp.nonzero()
            link_mask_tmp = np.zeros(link_mask_size, dtype=np.uint8)
            for j in range(link_mask_size[0]): # neighbors directions
                link_mask_tmp[j][weight_tmp_nonzero] = 1
            link_mask_shift = np.zeros(link_mask_size, dtype=np.uint8)
            w_index = weight_tmp_nonzero[1]
            w_index1 = w_index + 1
            w_index1[w_index1 >= link_mask_size[1]] = link_mask_size[1] - 1
            w_index_1 = w_index - 1
            w_index_1[w_index_1 < 0] = 0
            h_index = weight_tmp_nonzero[0]
            h_index1 = h_index + 1
            h_index1[h_index1 >= link_mask_size[2]] = link_mask_size[2] - 1
            h_index_1 = h_index - 1
            h_index_1[h_index_1 < 0] = 0
            link_mask_shift[0][h_index1, w_index1] = 1
            link_mask_shift[1][h_index1, w_index] = 1
            link_mask_shift[2][h_index1, w_index_1] = 1
            link_mask_shift[3][h_index1, w_index_1] = 1
            link_mask_shift[4][h_index_1, w_index_1] = 1
            link_mask_shift[5][h_index_1, w_index] = 1
            link_mask_shift[6][h_index_1, w_index1] = 1
            link_mask_shift[7][h_index, w_index1] = 1

            for j in range(link_mask_size[0]):
                # +0 to convert bool array to int array
                link_mask[j] += np.logical_and(link_mask_tmp[j], link_mask_shift[j]).astype(np.uint8)
        return torch.LongTensor(pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)

    @staticmethod
    def calc_pixel_weight_and_link_weight(true_pixel_mask, predict_pixel_mask, pixel_pos_weight):
        pass

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