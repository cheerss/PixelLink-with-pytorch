from torch.utils.data import Dataset
from PIL import Image
import config
import torch
import codecs
import random
import math
import copy
import time

class ICDAR15Dataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.all_images = self.read_datasets(images_dir, config.all_trains)
        self.all_labels = self.read_labels(labels_dir, config.all_trains)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        if isinstance(index, int):
            return {'image': self.all_images[index], 'label': all_labels[index]}

    def read_datasets(self, dir, num):
        res = []
        for i in range(1, num+1):
            res.append(Image.open(dir+ "img_" + str(i) + ".jpg"))
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
        return {'image': image, 'label': label}

    def data_transform(self, index):
        image = self.all_images[index]
        label = self.all_labels[index]

        rotate_rand = random.randint(0, 3)
        # rotate
        image = image.rotate(90 * rotate_rand, expand=True)
        image.save("rotate.jpg", "JPEG")
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
            new_h_start = (h - new_w) // 2
            new_w_start = (w - new_w) // 2
        # print("size after crop should be (h, w): (" + str(new_h) + " " + str(new_w) + ")")
        # print("start after crop(h, w): (" + str(new_h_start) + " " + str(new_w_start) + ")")

        # crop
        image = image.crop((new_w_start, new_h_start, new_w_start+new_w, new_h_start+new_h))
        # print("size after crop(h, w): (" + str(image.size[1]) + " " + str(image.size[0]) + ")")
        image.save("crop.jpg", "JPEG")
        # resize
        image = image.resize((512, 512), resample=Image.BILINEAR)
        image.save("resize.jpg", "JPEG")
        # print("size after resize(h, w): (" + str(image.size[1]) + " " + str(image.size[0]) + ")")

        new_label = copy.deepcopy(label)
        # ground truth file: x along width, y along height
        box_areas = []
        for box in new_label:
                box[0], box[1] = box[1], box[0]
                box[2], box[3] = box[3], box[2]
                box[4], box[5] = box[5], box[4]
                box[6], box[7] = box[7], box[6]
                box_areas.append((box[3] - box[1]) * (box[6] - box[0]))
                # print("box original: (" + str(box[0]) + "," + str(box[1]) + ") (" +
                #                           str(box[2]) + "," + str(box[3]) + ") (" +
                #                           str(box[4]) + "," + str(box[5]) + ") (" + 
                #                           str(box[6]) + "," + str(box[7]) + ")")
        # label rotate
        for i in range(rotate_rand):
            for box in new_label:
                box[2], box[3] = box[1], h-1-box[0]
                box[4], box[5] = box[3], h-1-box[2]
                box[6], box[7] = box[5], h-1-box[4]
                box[0], box[1] = box[7], h-1-box[6]
        # for box in new_label:
            # print("box after rotate: (" + str(box[0]) + "," + str(box[1]) + ") (" +
            #                               str(box[2]) + "," + str(box[3]) + ") (" +
            #                               str(box[4]) + "," + str(box[5]) + ") (" + 
            #                               str(box[6]) + "," + str(box[7]) + ")")
        # label crop
        for box in new_label:
            for i in range(4):
                box[2*i] -= new_h_start
                box[2*i+1] -= new_w_start
                if(box[2*i] < 0):
                    box[2*i] = 0
                if(box[2*i+1] < 0):
                    box[2*i+1] = 0
            # print("box after crop: (" + str(box[0]) + "," + str(box[1]) + ") (" +
            #                             str(box[2]) + "," + str(box[3]) + ") (" +
            #                             str(box[4]) + "," + str(box[5]) + ") (" + 
            #                             str(box[6]) + "," + str(box[7]) + ")")
        # label resize
        for box in new_label:
            for i in range(4):
                box[2*i] = int(box[2*i] * 512 / new_h)
                box[2*i+1] = int(box[2*i+1] * 512 / new_w)
            # print("box after resize: (" + str(box[0]) + "," + str(box[1]) + ") (" +
            #                               str(box[2]) + "," + str(box[3]) + ") (" +
            #                               str(box[4]) + "," + str(box[5]) + ") (" + 
            #                               str(box[6]) + "," + str(box[7]) + ")")

        # delete the boxes which unsatisfy the conditions
        for i in range(len(new_label)-1, -1, -1):
            box = new_label[i]
            min_side = min(box[3] - box[1], box[6] - box[0])

            if min_side < 10:
                del new_label[i]
                continue
            new_box_area = (box[3] - box[1]) * (box[6] - box[0])
            if new_box_area / box_areas[i] < 0.2:
                del new_label[i]
        return image, new_label

    @staticmethod
    def label_to_pixel_mask(label, img_size, version="2s"):
        factor = 2
        if version == "4s":
            factor = 4
        pixel_mask = torch.zeros(img_size / factor, dtype=torch.int8)
        for box in label:
            a_x = (int)((box[0] + box[2]) / 2 / factor)
            b_x = (int)((box[4] + box[6]) / 2 / factor)
            a_y = (int)((box[1] + box[7]) / 2 / factor)
            b_y = (int)((box[3] + box[5]) / 2 / factor)
            pixel_mask[a_x: b_x, a_y: b_y] = 1
        return pixel_mask

    @staticmethod
    def label_to_pixel_weight(label, img_size, version="2s"):
        """
        Return: torch.FloatTensor(img_size / factor)
        """
        factor = 2
        if version == "4s":
            factor = 4
        pixel_weight = torch.zeros(img_size / factor, dtype=torch.float)
        # no box
        if len(label) == 0:
            return pixel_weight
        label_areas = []
        for box in label:
            label_areas.append((box[4] - box[0]) * (box[5] - box[1]) / factor ** 2)
        # all_areas = sum(label_areas)
        avg_weight = sum(label_areas) / len(label)
        for i in range(len(label)):
            box = label[i]
            a_x = (int)((box[0] + box[2]) / 2 / factor)
            b_x = (int)((box[4] + box[6]) / 2 / factor)
            a_y = (int)((box[1] + box[7]) / 2 / factor)
            b_y = (int)((box[3] + box[5]) / 2 / factor)
            pixel_weight[a_x: b_x, a_y: b_y] = avg_weight / label_areas[i]
        return pixel_weight

    @staticmethod
    def label_to_link_mask(label, img_size, version="2s", neighbors=8):
        """
        8 neighbors:
            0 1 2
            7 - 3
            6 5 4
        Return: torch.FloatTensor(img_size / factor, neighbors)
        """
        factor = 2
        if version == "4s":
            factor = 4
        link_mask_size = list(img_size / factor).append(neighbors)
        link_mask = torch.zeros(link_mask_size, dtype=torch.int8)
        for box in label:
            a_x = (int)((box[0] + box[2]) / 2 / factor)
            b_x = (int)((box[4] + box[6]) / 2 / factor)
            a_y = (int)((box[1] + box[7]) / 2 / factor)
            b_y = (int)((box[3] + box[5]) / 2 / factor)
            link_mask[a_x + 1: b_x + 1, a_y + 1: b_y + 1, 0] = 1
            link_mask[a_x + 1: b_x + 1, a_y + 0: b_y + 1, 1] = 1
            link_mask[a_x + 1: b_x + 1, a_y + 0: b_y + 0, 2] = 1
            link_mask[a_x + 0: b_x + 1, a_y + 0: b_y + 0, 3] = 1
            link_mask[a_x + 0: b_x + 0, a_y + 0: b_y + 0, 4] = 1
            link_mask[a_x + 0: b_x + 0, a_y + 0: b_y + 1, 5] = 1
            link_mask[a_x + 0: b_x + 0, a_y + 1: b_y + 1, 6] = 1
            link_mask[a_x + 0: b_x + 1, a_y + 1: b_y + 1, 7] = 1
        return link_mask

    @staticmethod
    def label_to_link_weight():
        pass

if __name__ == '__main__':
    start = time.time()
    dataset = PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    end = time.time()
    print("time to read datasets: " + str(end - start)) # about 0.12s

    start = time.time()
    for i in range(1000):
        sample = dataset.__getitem__(i)
    end = time.time()
    print("time to get 1000 items: " + str(end - start)) # about 34s

    sample['image'].save("transform.jpg", "JPEG")
    i = 0
    for box in sample['label']:
        a_x = (int)((box[0] + box[2]) / 2)
        b_x = (int)((box[4] + box[6]) / 2)
        a_y = (int)((box[1] + box[7]) / 2)
        b_y = (int)((box[3] + box[5]) / 2)

        image = sample['image'].crop((a_y, a_x, b_y, b_x))
        image.save(str(i) + ".jpg", "JPEG")
        i += 1
