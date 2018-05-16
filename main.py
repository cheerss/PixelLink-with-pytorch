import net
import numpy as np
import torch
import torch.nn as nn
# import read_data
import datasets
from torch import optim
from criterion import PixelLinkLoss
import loss
import config
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

def main():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    my_net = net.Net()

    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
    else:
        device = torch.device("cpu")

    optimizer = optim.SGD(my_net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    for i in range(config.epoch):
        for i_batch, sample in enumerate(dataloader):
            images = sample['image'].to(device)
            pixel_masks = sample['pixel_mask'].to(device)
            link_masks = sample['link_mask'].to(device)
            pixel_pos_weights = sample['pixel_pos_weight'].to(device)

            out_1, out_2 = my_net.forward(images)
            loss_instance = PixelLinkLoss()

            pixel_loss = loss_instance.pixel_loss(out_1, pixel_masks, pixel_pos_weights)
            link_loss = loss_instance.link_loss(out_2, link_masks)
            losses = config.pixel_weight * pixel_loss + config.link_weight * link_loss
            print("epoch " + str(i) + " iteration " + str(i_batch) + ": ", end="")
            print("pixel_loss: " + str(pixel_loss.tolist()) + ", ", end="")
            print("link_loss: " + str(link_loss.tolist()) + ", ", end="") 
            print("total loss: " + str(losses.tolist()))
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (i_batch + 1) % 200 == 0:
                torch.save(my_net.state_dict(), config.saving_model_dir + str(i) + "_" + str(i_batch) + ".mdl")

if __name__ == "__main__":
    main()
