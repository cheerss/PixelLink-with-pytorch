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
import postprocess
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import os
import cv2
import time
import argparse

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='')
parser.add_argument('change', metavar='N', type=int,
                    help='an integer for change')
args = parser.parse_args()

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)

def test_res(image, pixel_out):
    """
    image size: H * W * C, 256
    pixel_out size: 2 * H * C, 256
    """
    pixel_out = pixel_out[1] > pixel_out[0]
    # import IPython
    # IPython.embed()
    pixel_out = pixel_out.unsqueeze(2).expand(-1, -1, 3)
    pixel_out = 1 - pixel_out

    return image * pixel_out

def test_model():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    my_net = net.Net()
    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")
    my_net.load_state_dict(torch.load(config.saving_model_dir + '24600.mdl'))

    batch = 0
    for i_batch, sample in enumerate(dataloader):
        images = sample['image'].to(device)
        out_1, out_2 = my_net.forward(images)
        rects = postprocess.mask_to_box(out_1, out_2)
        for i in range(config.batch_size):
            image = sample['image'][i].data.numpy() * 255
            pixel_out = out_1[i]
            image = image[(2, 1, 0), ...]
            image = np.transpose(image, (1, 2, 0))
            shape = image.shape
            image = image.reshape([int(shape[0]/2), 2, int(shape[1]/2), 2, shape[2]])
            image = image.max(axis=(1, 3))
            image = np.ascontiguousarray(image)
            image = test_res(image, pixel_out)
            # cv2.drawContours(image, rects[i], -1, (0, 255, 0))
            cv2.imwrite("res" + str(i) + ".jpg", image.cpu().numpy())
        batch += 1
        if batch > 5:
            break

def main():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    my_net = net.Net()

    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")

    # nn.init.xavier_uniform_(list(my_net.parameters()))
    my_net.apply(weight_init)
    optimizer = optim.SGD(my_net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    if args.change:
        optimizer2 = optim.SGD(my_net.parameters(), lr=config.learning_rate2, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        optimizer2 = optim.SGD(my_net.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    iteration = 0
    for i in range(config.epoch):
        for i_batch, sample in enumerate(dataloader):
            start = time.time()
            images = sample['image'].to(device)
            pixel_masks = sample['pixel_mask'].to(device)
            link_masks = sample['link_mask'].to(device)
            pixel_pos_weights = sample['pixel_pos_weight'].to(device)

            out_1, out_2 = my_net.forward(images)
            loss_instance = PixelLinkLoss()
            # print(out_2)

            pixel_loss = loss_instance.pixel_loss(out_1, pixel_masks, pixel_pos_weights)
            link_loss = loss_instance.link_loss(out_2, link_masks)
            losses = config.pixel_weight * pixel_loss + config.link_weight * link_loss
            print("epoch " + str(i) + " iteration " + str(i_batch), end=": ")
            print("pixel_loss: " + str(pixel_loss.tolist()), end=", ")
            print("link_loss: " + str(link_loss.tolist()), end=", ") 
            print("total loss: " + str(losses.tolist()), end=", ")
            if iteration < 200:
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            else:
                optimizer2.zero_grad()
                losses.backward()
                optimizer2.step()
            end = time.time()
            print("time: " + str(end - start))
            if (iteration + 1) % 200 == 0:
                if args.change:
                    saving_model_dir = config.saving_model_dir2
                else:
                    saving_model_dir = config.saving_model_dir
                torch.save(my_net.state_dict(), saving_model_dir + str(iteration + 1) + ".mdl")
            iteration += 1

if __name__ == "__main__":
    # main()
    test_model()
