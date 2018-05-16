import torch
import torch.nn as nn
import config

class PixelLinkLoss(object):
    def __init__(self):
        self.pixel_cross_entropy_layer = nn.CrossEntropyLoss(reduce=False)
        self.link_cross_entropy_layer = nn.CrossEntropyLoss(reduce=False)
        self.pixel_cross_entropy = None
        self.link_cross_entropy = None
        self.pixel_weight = None
        self.pos_link_weight = None
        self.neg_link_weight = None
        self.area = None

    def pixel_loss(self, input, target, pos_weight):
        batch_size = input.size(0)
        self.pixel_cross_entropy = self.pixel_cross_entropy_layer(input, target)
        self.area = torch.sum(target.view(batch_size, -1), dim=1)
        int_area = self.area.to(torch.int).data.tolist()
        self.area = self.area.to(torch.float)
        # input[0] is for negative
        for i in range(batch_size):
            wrong_input = input[i, 0][target[i]==0].view(-1)
            # print("k: " + str(int_area[i] * config.neg_pos_ratio))
            neg_area = min(int_area[i] * config.neg_pos_ratio, wrong_input.size(0))
            topk, _ = torch.topk(wrong_input, neg_area)
            self.pixel_weight = pos_weight
            self.pixel_weight[i][input[i, 0] > topk[-1]] = 1
        weighted_pixel_cross_entropy = self.pixel_weight * self.pixel_cross_entropy
        weighted_pixel_cross_entropy = weighted_pixel_cross_entropy.view(batch_size, -1)

        # import IPython
        # IPython.embed()
        return torch.mean(torch.sum(weighted_pixel_cross_entropy, dim=1) / ((1 + config.neg_pos_ratio) * self.area))

    def link_loss(self, input, target, neighbors=8):
        batch_size = input.size(0)
        self.pos_link_weight = (target == 1).to(torch.float) * \
            self.pixel_weight.unsqueeze(1).expand(-1,neighbors,-1,-1)
        self.neg_link_weight = (target == 0).to(torch.float) * \
            self.pixel_weight.unsqueeze(1).expand(-1,neighbors,-1,-1)
        sum_pos_link_weight = torch.sum(self.pos_link_weight.view(batch_size, -1), dim=1)
        sum_neg_link_weight = torch.sum(self.neg_link_weight.view(batch_size, -1), dim=1)
        self.link_cross_entropy = self.pos_link_weight.new_empty(self.pos_link_weight.size())
        for i in range(neighbors):
            this_input = input[:, [2*i, 2*i+1]]
            this_target = target[:, i].squeeze(1)
            self.link_cross_entropy[:, i] = self.link_cross_entropy_layer(this_input, this_target)
        loss_link_pos = self.pos_link_weight.new_empty(self.pos_link_weight.size())
        loss_link_neg = self.neg_link_weight.new_empty(self.neg_link_weight.size())
        for i in range(batch_size):
            loss_link_pos[i] = self.pos_link_weight[i] * self.link_cross_entropy[i] / sum_pos_link_weight[i]
            loss_link_neg[i] = self.neg_link_weight[i] * self.link_cross_entropy[i] / sum_neg_link_weight[i]
        loss_link_pos = torch.sum(loss_link_pos.view(batch_size, -1), dim=1)
        loss_link_neg = torch.sum(loss_link_neg.view(batch_size, -1), dim=1)
        return torch.mean(loss_link_pos + loss_link_neg)