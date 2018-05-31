import torch
import torch.nn as nn
import config

class PixelLinkLoss(object):
    def __init__(self):
        self.pixel_cross_entropy_layer = nn.CrossEntropyLoss(reduce=False)
        self.link_cross_entropy_layer = nn.CrossEntropyLoss(reduce=False)
        # self.pixel_cross_entropy_layer = nn.BCELoss(reduce=False)
        # self.link_cross_entropy_layer = nn.BCELoss(reduce=False)
        self.softmax_layer = nn.Softmax2d()
        # self.softmax_layer = nn.LogSoftmax(1)
        self.pixel_cross_entropy = None
        self.link_cross_entropy = None
        self.pos_pixel_weight = None
        self.neg_pixel_weight = None
        self.pixel_weight = None
        self.pos_link_weight = None
        self.neg_link_weight = None
        self.area = None
        self.neg_area = None

    def pixel_loss(self, input, target, neg_pixel_masks, pos_weight, stop):
        batch_size = input.size(0)
        softmax_input = self.softmax_layer(input)
        if stop:
            import IPython
            IPython.embed()
        self.pixel_cross_entropy = self.pixel_cross_entropy_layer(input, target)
        self.area = torch.sum(target.view(batch_size, -1), dim=1)
        int_area = self.area.to(torch.int).data.tolist()
        self.area = self.area.to(torch.float)
        # input[0] is for negative
        self.pos_pixel_weight = pos_weight
        self.neg_pixel_weight = torch.zeros_like(self.pos_pixel_weight, dtype=torch.uint8)
        self.neg_area = torch.zeros_like(self.area, dtype=torch.int)
        for i in range(batch_size):
            # wrong_input = softmax_input[i, 0][target[i]==0].view(-1)
            wrong_input = softmax_input[i, 0][neg_pixel_masks[i]==1].view(-1)
            # print("k: " + str(int_area[i] * config.neg_pos_ratio))
            r_pos_area = int_area[i] * config.neg_pos_ratio
            if r_pos_area == 0:
                r_pos_area = 10000
            self.neg_area[i] = min(r_pos_area, wrong_input.size(0))
            # the smaller the wrong_input is, the bigger the loss is
            topk, _ = torch.topk(-wrong_input, self.neg_area[i].tolist()) # top_k is negative
            self.neg_pixel_weight[i][softmax_input[i, 0] <= -topk[-1]] = 1
            self.neg_pixel_weight[i] = self.neg_pixel_weight[i] & (neg_pixel_masks[i]==1)
            # print("neg area should be %d" % self.neg_area[i].tolist(), end=", ")
            # print("neg area is %d" % self.neg_pixel_weight[i].sum().tolist())
        # print("pos weight %f" % torch.sum(self.pos_pixel_weight).tolist(), end="")
        # print("neg weight %f" % torch.sum(self.neg_pixel_weight).tolist())
        self.pixel_weight = self.pos_pixel_weight + self.neg_pixel_weight.to(torch.float)
        weighted_pixel_cross_entropy_pos = self.pos_pixel_weight * self.pixel_cross_entropy
        weighted_pixel_cross_entropy_pos = weighted_pixel_cross_entropy_pos.view(batch_size, -1)

        weighted_pixel_cross_entropy_neg = self.neg_pixel_weight.to(torch.float) * self.pixel_cross_entropy
        weighted_pixel_cross_entropy_neg = weighted_pixel_cross_entropy_neg.view(batch_size, -1)
        weighted_pixel_cross_entropy = weighted_pixel_cross_entropy_neg + weighted_pixel_cross_entropy_pos

        return [torch.mean(torch.sum(weighted_pixel_cross_entropy_pos, dim=1) / \
                (self.area + self.neg_area.to(torch.float))),
                torch.mean(torch.sum(weighted_pixel_cross_entropy_neg, dim=1) / \
                (self.area + self.neg_area.to(torch.float))),
                ]

    def link_loss(self, input, target, neighbors=8):
        batch_size = input.size(0)
        self.pos_link_weight = (target == 1).to(torch.float) * \
            self.pos_pixel_weight.unsqueeze(1).expand(-1, neighbors, -1, -1)
        self.neg_link_weight = (target == 0).to(torch.float) * \
            self.pos_pixel_weight.unsqueeze(1).expand(-1, neighbors, -1 ,-1)
        sum_pos_link_weight = torch.sum(self.pos_link_weight.view(batch_size, -1), dim=1)
        sum_neg_link_weight = torch.sum(self.neg_link_weight.view(batch_size, -1), dim=1)

        self.link_cross_entropy = self.pos_link_weight.new_empty(self.pos_link_weight.size())
        for i in range(neighbors):
            assert input.size(1) == 16
            # input = input.contiguous()
            this_input = input[:, [2 * i, 2 * i + 1]]
            # this_input = self.softmax_layer(this_input)
            # import IPython
            # IPython.embed()
            this_target = target[:, i].squeeze(1)
            # this_target = this_target.contiguous()
            # assert this_input.is_contiguous()
            # assert this_target.is_contiguous()
            # print(this_target)
            # print(torch.sum(this_target>=2))
            self.link_cross_entropy[:, i] = self.link_cross_entropy_layer(this_input, this_target)
        loss_link_pos = self.pos_link_weight.new_empty(self.pos_link_weight.size())
        loss_link_neg = self.neg_link_weight.new_empty(self.neg_link_weight.size())
        for i in range(batch_size):
            if sum_pos_link_weight[i].tolist() == 0:
                loss_link_pos[i] = 0
            else:
                loss_link_pos[i] = self.pos_link_weight[i] * self.link_cross_entropy[i] / sum_pos_link_weight[i]
            if sum_neg_link_weight[i].tolist() == 0:
                loss_link_neg[i] = 0
            else:
                loss_link_neg[i] = self.neg_link_weight[i] * self.link_cross_entropy[i] / sum_neg_link_weight[i]
        loss_link_pos = torch.sum(loss_link_pos.view(batch_size, -1), dim=1)
        loss_link_neg = torch.sum(loss_link_neg.view(batch_size, -1), dim=1)
        # import IPython
        # IPython.embed()
        return torch.mean(loss_link_pos), torch.mean(loss_link_neg)