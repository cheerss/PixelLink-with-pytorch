import torch
import cv2
import numpy as np
import torch.nn as nn

def mask_filter(pixel_mask, link_mask, neighbors=8, scale=4):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    pixel_class = nn.Softmax2d()(pixel_mask)
    # print(pixel_class.shape)
    pixel_class = pixel_class[:, 1] > 0.4
    # pixel_class = pixel_mask[:, 1] > pixel_mask[:, 0]
    # link_neighbors = torch.ByteTensor([batch_size, neighbors, mask_height, mask_width])
    link_neighbors = torch.zeros([batch_size, neighbors, mask_height, mask_width], \
                                    dtype=torch.uint8, device=pixel_mask.device)
    
    for i in range(neighbors):
        # print(link_mask[:, [2 * i, 2 * i + 1]].shape)
        tmp = nn.Softmax2d()(link_mask[:, [2 * i, 2 * i + 1]])
        # print(tmp.shape)
        link_neighbors[:, i] = tmp[:, 1] > 0.4
        # link_neighbors[:, i] = link_mask[:, 2 * i + 1] > link_mask[:, 2 * i] 
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class[i]
    # res_mask = np.zeros([batch_size, mask_height, mask_width], dtype=np.uint8)
    return pixel_class, link_neighbors

def mask_to_box(pixel_mask, link_mask, neighbors=8, scale=4):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    pixel_class = nn.Softmax2d()(pixel_mask)
    # print(pixel_class.shape)
    pixel_class = pixel_class[:, 1] > 0.4
    # pixel_class = pixel_mask[:, 1] > pixel_mask[:, 0]
    # link_neighbors = torch.ByteTensor([batch_size, neighbors, mask_height, mask_width])
    link_neighbors = torch.zeros([batch_size, neighbors, mask_height, mask_width], \
                                    dtype=torch.uint8, device=pixel_mask.device)
    
    for i in range(neighbors):
        # print(link_mask[:, [2 * i, 2 * i + 1]].shape)
        tmp = nn.Softmax2d()(link_mask[:, [2 * i, 2 * i + 1]])
        # print(tmp.shape)
        link_neighbors[:, i] = tmp[:, 1] > 0.4
        # link_neighbors[:, i] = link_mask[:, 2 * i + 1] > link_mask[:, 2 * i] 
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class[i]
    # res_mask = np.zeros([batch_size, mask_height, mask_width], dtype=np.uint8)
    all_boxes = []
    for i in range(batch_size):
        res_mask = func(pixel_class[i], link_neighbors[i])
        box_num = np.amax(res_mask)
        # print(res_mask.any())
        bounding_boxes = []
        for i in range(box_num):
            box_mask = (res_mask == i).astype(np.uint8)
            # if box_mask.sum() < 10:
                # continue
            box_mask, contours, _ = cv2.findContours(box_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            # print(contours[0])
            bounding_box = cv2.minAreaRect(contours[0])
            bounding_box = cv2.boxPoints(bounding_box)
            bounding_box = np.clip(bounding_box * scale, 0, 128 * scale - 1).astype(np.int)
            # import IPython
            # IPython.embed()
            bounding_boxes.append(bounding_box)
        all_boxes.append(bounding_boxes)
    return all_boxes

def get_neighbors(h_index, w_index):
    res = []
    res.append((h_index - 1, w_index - 1))
    res.append((h_index - 1, w_index))
    res.append((h_index - 1, w_index + 1))
    res.append((h_index, w_index + 1))
    res.append((h_index + 1, w_index + 1))
    res.append((h_index + 1, w_index))
    res.append((h_index + 1, w_index - 1))
    res.append((h_index, w_index - 1))
    return res

def func(pixel_cls, link_cls):
    def joint(pointa, pointb):
        roota = find_root(pointa)
        rootb = find_root(pointb)
        if roota != rootb:
            group_mask[rootb] = roota
            # group_mask[pointb] = roota
            # group_mask[pointa] = roota
        return

    def find_root(pointa):
        root = pointa
        while group_mask.get(root) != -1:
            root = group_mask.get(root)
        return root

    pixel_cls = pixel_cls.cpu().numpy()
    link_cls = link_cls.cpu().numpy()

    # import IPython
    # IPython.embed()

    # print(pixel_cls.any())
    # print(np.where(pixel_cls))
    pixel_points = list(zip(*np.where(pixel_cls)))
    h, w = pixel_cls.shape
    group_mask = dict.fromkeys(pixel_points, -1)
    # print(group_mask)

    for point in pixel_points:
        h_index, w_index = point
        # print(point)
        neighbors = get_neighbors(h_index, w_index)
        for i, neighbor in enumerate(neighbors):
            nh_index, nw_index = neighbor
            if nh_index < 0 or nw_index < 0 or nh_index >= h or nw_index >= w:
                continue
            if pixel_cls[nh_index, nw_index] == 1 and link_cls[i, h_index, w_index] == 1:
                joint(point, neighbor)

    res = np.zeros(pixel_cls.shape, dtype=np.uint8)
    root_map = {}
    for point in pixel_points:
        h_index, w_index = point
        root = find_root(point)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        res[h_index, w_index] = root_map[root]

    return res

