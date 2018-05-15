import torch.nn as nn
import numpy as np
import torch
import config

def loss_calculator(out_1, out_2, pixel_masks, link_masks, pixel_weights):
    link_weight = config.link_weight
    pixel_weight = config.pixel_weight
    loss = pixel_weight * pixel_loss(out_1, pixel_masks, pixel_weights) + \
           link_weight * link_loss(out_2, link_masks)
    return loss

def pixel_loss(pixel_outputs, pixel_masks, pixel_weights):
    r = config.neg_pos_ratio
    loss_pixel = nn.functional.cross_entropy(pixel_outputs, pixel_masks, reduce=False)
    postive_areas = torch.Tensor([torch.nonzero(pixel_mask).size(0) for pixel_mask in pixel_masks]).cuda()
    weights = pixel_weights + OHDM_weights(pixel_outputs, pixel_masks)
    print("pixel_loss")
    return torch.sum(torch.sum(torch.sum(loss_pixel * weights, 2), 1) / ((1 + r) * postive_areas)) / pixel_masks.size(0)

def link_loss(link_output, link_masks):
    return 0

def instance_balanced_weights(labels, pixel_masks, postive_areas):
    num_of_boxes = np.array([len(boxes) for boxes in labels])
    weights_each_box = postive_areas / num_of_boxes
    weights = np.zeros_like(pixel_masks)
    batch_size = pixel_masks.size(0)
    for i in range(batch_size):
        for box in labels[i]:
            a_x = (int)((box[0] + box[6]) / 2)
            b_x = (int)((box[2] + box[4]) / 2)
            a_y = (int)((box[1] + box[7]) / 2)
            b_y = (int)((box[3] + box[5]) / 2)
            s = (b_x - a_x + 1) * (b_y - a_y + 1)
            weights[i][a_x: b_x, a_y: b_y] = weights_each_box[i] / s
    return weights


def OHDM_weights(out, pixel_masks):
    weights = pixel_masks.new_zeros(pixel_masks.size(), dtype=torch.float32)
    return weights
