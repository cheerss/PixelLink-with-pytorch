import postprocess
import cv2
import numpy as np
import config
import datasets
import torch
from torch.utils.data import DataLoader
import net
import torch.nn as nn
import ImgLib.ImgFormat as ImgFormat
import ImgLib.ImgTransform as ImgTransform

def cal_label_on_batch(my_net, imgs):
    scale = 2 if config.version == "2s" else 4
    with torch.no_grad():
        out_1, out_2 = my_net.forward(imgs)
    all_boxes = postprocess.mask_to_box(out_1, out_2, scale=scale)
    return all_boxes

def cal_IOU(box1, box2):
    """
    box1, box2: list or numpy array of size 4*2 or 8, h_index first
    """
    box1 = np.array(box1).reshape([1, 4, 2])
    box2 = np.array(box2).reshape([1, 4, 2])
    box1_max = box1.max(axis=1)
    box2_max = box2.max(axis=1)
    w_max = max(box1_max[0][0], box2_max[0][0])
    h_max = max(box1_max[0][1], box2_max[0][1])
    canvas = np.zeros((h_max + 1, w_max + 1))
    # print(canvas.shape)
    box1_canvas = canvas.copy()
    box1_area = np.sum(cv2.drawContours(box1_canvas, box1, -1, 1, thickness=-1))
    # print(box1_area)
    box2_canvas = canvas.copy()
    box2_area = np.sum(cv2.drawContours(box2_canvas, box2, -1, 1, thickness=-1))
    # print(box2_area)
    cv2.drawContours(canvas, box1, -1, 1, thickness=-1)
    cv2.drawContours(canvas, box2, -1, 1, thickness=-1)
    union = np.sum(canvas)
    # print(union)
    intersction = box1_area + box2_area - union
    return intersction / union

def comp_gt_and_output(my_labels, gt_labels, threshold):
    """
    return: [true_pos, false_pos, false_neg]
    """
    coor = gt_labels["coor"]
    ignore = gt_labels["ignore"]
    true_pos, true_neg, false_pos, false_neg = [0] * 4
    for my_label in my_labels:
        for gt_label in coor:
            if cal_IOU(my_label, gt_label) > threshold:
                true_pos += 1
                break
        else:
            false_pos += 1
    for i, gt_label in enumerate(coor):
        if ignore[i]:
            continue
        for my_label in my_labels:
            if cal_IOU(gt_label, my_label) > threshold:
                break
        else:
            false_neg += 1
    return true_pos, false_pos, false_neg

def test_on_train_dataset(vis_per_img=10):
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir, train=False)
    # dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    my_net = net.Net()
    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        if config.multi_gpu:
            my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")
    my_net.load_state_dict(torch.load(config.saving_model_dir + '%d.mdl' % config.test_model_index))
    true_pos, true_neg, false_pos, false_neg = [0] * 4
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image'].to(device)
        image = image.unsqueeze(0)
        my_labels = cal_label_on_batch(my_net, image)[0]
        # print("my labels num: %d" % len(my_labels))
        res = comp_gt_and_output(my_labels, sample["label"], 0.5)
        if i % vis_per_img == 0:
            image = image.squeeze(0).cpu().numpy()
            image = ImgFormat.ImgOrderFormat(image, from_order="CHW", to_order="HWC")
            image = ImgTransform.UnzeroMeanImage(image, config.r_mean, config.g_mean, config.b_mean)
            image = ImgFormat.ImgColorFormat(image, from_color="RGB", to_color="BGR")
            image = visualize_label(image, my_labels, color=(0, 255, 0))
            image = visualize_label(image, sample["label"]["coor"], color=(255, 0, 0))
            cv2.imwrite("img_%d.jpg" % i, image)
        true_pos += res[0]
        false_pos += res[1]
        false_neg += res[2]
        if (true_pos + false_pos) > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0
        if (true_pos + false_neg) > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = 0
        print("i: %d, TP: %d, FP: %d, FN: %d, precision: %f, recall: %f" % (i, true_pos, false_pos, false_neg, precision, recall))

def visualize_label(img, boxes, color=(0, 255, 0)):
    """
    img: HWC
    boxes: array of num * 4 * 2
    """
    boxes = np.array(boxes).reshape(-1, 4, 2)
    img = np.ascontiguousarray(img)
    cv2.drawContours(img, boxes, -1, color, thickness=1)
    return img