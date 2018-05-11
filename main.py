import net
import numpy as np
import torch
import torch.nn as nn
import read_data
import torch.optim
import loss
import config

learning_rate = config.learning_rate
all_trains = config.all_trains
batch_size = config.batch_size
momentum = config.momentum
weight_decay = config.weight_decay
iterations = config.iterations
gpu = config.gpu
pixel_weight = config.pixel_weight
link_weight = config.link_weight

def main():
    images = read_data.read_datasets(config.train_images_dir, all_trains)
    labels = read_data.read_ground_truth(config.train_labels_dir, all_trains)
    pixel_masks = read_data.trans_all_to_mask(labels)
    link_masks = read_data.trans_all_to_link_mask(labels)

    images = torch.Tensor(images)
    pixel_masks = torch.Tensor(pixel_masks)
    # labels = torch.LongTensor(labels)
    link_masks = torch.Tensor(link_masks)
    postive_areas = np.array([torch.nonzero(pixel_mask).size(0) for pixel_mask in pixel_masks])
    pixel_weights = loss.instance_balanced_weights(labels, pixel_masks, postive_areas)
    pixel_weights = torch.Tensor(pixel_weights)
    my_net = net.Net()
    pixel_masks = nn.functional.max_pool2d(pixel_masks, 2)
    pixel_masks = pixel_masks.type(torch.LongTensor)
    link_masks = nn.functional.max_pool2d(link_masks, 2)
    link_masks = link_masks.type(torch.LongTensor)
    pixel_weights = nn.functional.max_pool2d(pixel_weights, 2)

    if gpu:
        my_net = my_net.cuda()
        images = images.cuda()
        # labels = labels.cuda()
        pixel_masks = pixel_masks.cuda()
        link_masks = link_masks.cuda()
        pixel_weights = pixel_weights.cuda()
        optimizer = torch.optim.SGD(my_net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    for i in range(iterations):
        print("iterations:" + str(i))
        images_index = torch.randperm(all_trains)
        images_index = images_index[:batch_size]
        data = images[images_index]
        tmp_pixel_masks = pixel_masks[images_index]
        tmp_link_masks = link_masks[images_index]
        tmp_pixel_weights = pixel_weights[images_index]
        # tmp_labels = labels[images_index]
        out_1, out_2 = my_net.forward(data)

        losses = loss.loss_calculator(out_1, out_2, tmp_pixel_masks, tmp_link_masks, tmp_pixel_weights)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

if __name__ == "__main__":
    main()
