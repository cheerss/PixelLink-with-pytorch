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
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
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
        nn.init.constant_(m.bias.data, 0)

def test_res1(image, pixel_out, link_out):
    """
    image size: H * W * C, 256
    pixel_out size: 2 * H * C, 256
    link_out size: 16 * H * C, 256
    """
    pixel_out = pixel_out[1] > pixel_out[0]
    link_res = torch.ones_like(pixel_out)
    for i in range(8):
        link_res = link_res & (link_out[2 * i + 1] > link_out[2 * i])
    # import IPython
    # IPython.embed()
    pixel_out = pixel_out & link_res
    pixel_out = pixel_out.unsqueeze(2).expand(-1, -1, 3)
    # pixel_out = 1 - pixel_out

    return image * (1 - pixel_out)

def test_res():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir, train=False)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    my_net = net.Net()
    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        if config.multi_gpu:
            my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")
    my_net.load_state_dict(torch.load(config.saving_model_dir3 + '156600.mdl'))

    batch = 0
    for i_batch, sample in enumerate(dataloader):
        images = sample['image'].to(device)
        with torch.no_grad():
            out_1, out_2 = my_net.forward(images)
        pixel_masks, link_masks = postprocess.mask_filter(out_1, out_2)
        pixel_masks = np.repeat(np.expand_dims(pixel_masks, axis=3), 3, axis=3)
        # link_masks = np.repeat(link_mask, 3, axis=1)

        for i in range(1):
            pixel_mask = pixel_masks[i]
            link_mask = link_masks[i]
            image = sample['image'][i].data.numpy()
            label = dataset.get_label(i_batch * config.batch_size + i)

            image[0] += config.r_mean
            image[1] += config.g_mean
            image[2] += config.b_mean
            image = image[(2, 1, 0), ...]
            image = np.transpose(image, (1, 2, 0))
            shape = image.shape
            image = image.reshape([int(shape[0]/4), 4, int(shape[1]/4), 4, shape[2]])
            image = image.max(axis=(1, 3))

            img = image.copy()
            # import IPython
            # IPython.embed()
            # img[pixel_mask == 1] = 0
            img = img * (1 - pixel_mask)
            img = np.ascontiguousarray(img)
            cv2.imwrite("res_%d_pixel.jpg" % i, img)

            for j in range(8):
                img = image.copy()
                link = np.repeat(np.expand_dims(link_mask[j], axis=2), 3, axis=2)
                # img[link == 1] = 0
                img = img * (1 - link)
                img = np.ascontiguousarray(img)
                cv2.imwrite("res_%d_link_%d.jpg" % (i, j), img)
        batch += 1
        if batch >= 1:
            break

def test_model():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir, train=False)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    my_net = net.Net()
    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        if config.multi_gpu:
            my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")
    my_net.load_state_dict(torch.load(config.saving_model_dir3 + '156600.mdl'))

    batch = 0
    for i_batch, sample in enumerate(dataloader):
        images = sample['image'].to(device)
        with torch.no_grad():
            out_1, out_2 = my_net.forward(images)
        all_boxes = postprocess.mask_to_box(out_1, out_2)

        for i in range(config.batch_size):
        # for i in range(1):
            image = sample['image'][i].data.numpy()
            label = dataset.get_label(i_batch * config.batch_size + i)
            pixel_out = out_1[i]
            link_out = out_2[i]
            image[0] += config.r_mean
            image[1] += config.g_mean
            image[2] += config.b_mean
            image = image[(2, 1, 0), ...]
            image = np.transpose(image, (1, 2, 0))
            shape = image.shape
            # image = image.reshape([int(shape[0]/4), 4, int(shape[1]/4), 4, shape[2]])
            # image = image.max(axis=(1, 3))
            image = np.ascontiguousarray(image)
            # image = test_res(image, pixel_out, link_out)
            # all_boxes = postprocess.mask_to_box(pixel_out, link_out)
            # print(all_boxes[i])
            # for j in range(len(all_boxes[i])):
            #     all_boxes[i][j] = all_boxes[i][j].tolist()
            # print(all_boxes[i])
            all_boxes[i] = np.array(all_boxes[i])
            cv2.drawContours(image, all_boxes[i], -1, (0, 255, 0), thickness=1)
            cv2.drawContours(image, label, -1, (255, 0, 0), thickness=1)
            cv2.imwrite("res" + str(i) + ".jpg", image)
            print(i)
        batch += 1
        if batch >= 1:
            break

def retrain():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    sampler = WeightedRandomSampler([1/len(dataset)]*len(dataset), config.batch_size, replacement=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    my_net = net.Net()
    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        if config.multi_gpu:
            my_net = nn.DataParallel(my_net)
    else:
        device = torch.device("cpu")
    my_net.load_state_dict(torch.load(config.saving_model_dir3 + '119400.mdl'))
    optimizer = optim.SGD(my_net.parameters(), lr=config.retrain_learning_rate2, \
                            momentum=config.momentum, weight_decay=config.weight_decay)
    optimizer2 = optim.SGD(my_net.parameters(), lr=config.retrain_learning_rate, \
                            momentum=config.momentum, weight_decay=config.weight_decay)
    train(config.retrain_epoch, 119400, dataloader, my_net, optimizer, optimizer2, device)


def train(epoch, iteration, dataloader, my_net, optimizer, optimizer2, device):
    for i in range(epoch):
        for i_batch, sample in enumerate(dataloader):
            start = time.time()
            images = sample['image'].to(device)
            # print(images.shape, end=" ")
            pixel_masks = sample['pixel_mask'].to(device)
            neg_pixel_masks = sample['neg_pixel_mask'].to(device)
            link_masks = sample['link_mask'].to(device)
            pixel_pos_weights = sample['pixel_pos_weight'].to(device)

            out_1, out_2 = my_net.forward(images)
            loss_instance = PixelLinkLoss()
            # print(out_2)
            stop = False
            if iteration > 50:
                stop = False

            pixel_loss_pos, pixel_loss_neg = loss_instance.pixel_loss(out_1, pixel_masks, neg_pixel_masks, pixel_pos_weights, stop)
            pixel_loss = pixel_loss_pos + pixel_loss_neg
            link_loss_pos, link_loss_neg = loss_instance.link_loss(out_2, link_masks)
            link_loss = link_loss_pos + link_loss_neg
            losses = config.pixel_weight * pixel_loss + config.link_weight * link_loss
            print("iteration %d" % iteration, end=": ")
            print("pixel_loss: " + str(pixel_loss.tolist()), end=", ")
            print("pixel_loss_pos: " + str(pixel_loss_pos.tolist()), end=", ")
            print("pixel_loss_neg: " + str(pixel_loss_neg.tolist()), end=", ")
            print("link_loss: " + str(link_loss.tolist()), end=", ")
            # print("link_loss_pos: " + str(link_loss_pos.tolist()), end=", ")
            # print("link_loss_neg: " + str(link_loss_neg.tolist()), end=", ")
            print("total loss: " + str(losses.tolist()), end=", ")
            if iteration < 100:
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
                    saving_model_dir = config.saving_model_dir3
                else:
                    saving_model_dir = config.saving_model_dir
                torch.save(my_net.state_dict(), saving_model_dir + str(iteration + 1) + ".mdl")
            iteration += 1

def main():
    dataset = datasets.PixelLinkIC15Dataset(config.train_images_dir, config.train_labels_dir)
    sampler = WeightedRandomSampler([1/len(dataset)]*len(dataset), config.batch_size, replacement=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    # dataloader = DataLoader(dataset, config.batch_size, shuffle=True)
    my_net = net.Net()

    if config.gpu:
        device = torch.device("cuda:0")
        my_net = my_net.cuda()
        if config.multi_gpu:
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
    train(config.epoch, iteration, dataloader, my_net, optimizer, optimizer2, device)

if __name__ == "__main__":
    # main()
    # retrain()
    test_model()
    # test_res()
