import matplotlib.pylab as plt
import numpy as np
import codecs
import config

image_height = config.image_height
image_width = config.image_width
image_channel = config.image_channel

def read_image(filename):
    im = plt.imread(filename)
    return im

def read_datasets(dir, num, order="chw"):
    res = np.zeros((num, image_height, image_width, image_channel))
    for i in range(1, num+1):
        res[i-1] = plt.imread(dir+ "img_" + str(i) + ".jpg")
        if i % 100 == 0:
            print(i)
    if order == "chw":
        # channel, height, width
        res = res.transpose(0, 3, 1, 2)
    # print(res[0].shape)
    return res

def read_ground_truth(dir, num):
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

def trans_ground_truth_to_mask(boxes, shape, res):
    for box in boxes:
        assert(len(box)== 8)
        a_x = (int)((box[0] + box[6]) / 2)
        b_x = (int)((box[2] + box[4]) / 2)
        a_y = (int)((box[1] + box[7]) / 2)
        b_y = (int)((box[3] + box[5]) / 2)
        res[a_x: b_x + 1, a_y: b_y + 1] = 1
    return res

def trans_all_to_mask(labels, shape=[image_height, image_width]):
    res = np.zeros([len(labels),] + shape)
    i = 0
    for boxes in labels:
        trans_ground_truth_to_mask(boxes, shape, res[i])
        i += 1
    return res

def trans_boxes_to_link_mask(boxes, res, shape=[image_height, image_width], channels=8, neighbors=[0,1,2,3,4,5,6,7]):
    # assert(len(neighbors) == channels)
    # shape = [channels,] + shape
    # res = np.zeros(shape)
    # bias = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    for box in boxes:
        a_x = (int)((box[0] + box[6]) / 2)
        b_x = (int)((box[2] + box[4]) / 2)
        a_y = (int)((box[1] + box[7]) / 2)
        b_y = (int)((box[3] + box[5]) / 2)
        res[0][a_x+1: b_x+1, a_y+1: b_y+1] = 1
        res[1][a_x+1: b_x+1, a_y+0: b_y+1] = 1
        res[2][a_x+1: b_x+1, a_y+0: b_y+0] = 1
        res[3][a_x+0: b_x+1, a_y+0: b_y+0] = 1
        res[4][a_x+0: b_x+0, a_y+0: b_y+0] = 1
        res[5][a_x+0: b_x+0, a_y+0: b_y+1] = 1
        res[6][a_x+0: b_x+0, a_y+1: b_y+1] = 1
        res[7][a_x+0: b_x+1, a_y+1: b_y+1] = 1

def trans_all_to_link_mask(labels, shape=[image_height, image_width], channels=8, neighbors=[0,1,2,3,4,5,6,7]):
    assert len(neighbors) == channels
    res = np.zeros([len(labels), channels] + shape)
    i = 0
    for boxes in labels:
        trans_boxes_to_link_mask(boxes, res[i], shape, channels, neighbors)
        i += 1
    return res

if __name__ == "__main__":
    a = read_ground_truth("train_images/ground_truth/", 2)
    print(a)


