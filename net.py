import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: modify padding
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)

        self.out1_1 = nn.Conv2d(128, 2, 1, stride=1, padding=0)
        self.out1_2 = nn.Conv2d(128, 16, 1, stride=1, padding=0)
        self.out2_1 = nn.Conv2d(256, 2, 1, stride=1, padding=0)
        self.out2_2 = nn.Conv2d(256, 16, 1, stride=1, padding=0)
        self.out3_1 = nn.Conv2d(512, 2, 1, stride=1, padding=0)
        self.out3_2 = nn.Conv2d(512, 16, 1, stride=1, padding=0)
        self.out4_1 = nn.Conv2d(512, 2, 1, stride=1, padding=0)
        self.out4_2 = nn.Conv2d(512, 16, 1, stride=1, padding=0)
        self.out5_1 = nn.Conv2d(1024, 2, 1, stride=1, padding=0)
        self.out5_2 = nn.Conv2d(1024, 16, 1, stride=1, padding=0)

        self.final_1 = nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.final_2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)

    def forward(self, x):
        # print("forward1")
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        # print("forward11")
        x = self.conv2_2(self.conv2_1(x))
        # print("forward12")
        l1_1x = self.out1_1(x)
        # print("forward13")
        l1_2x = self.out1_2(x)
        # print("forward14")
        x = self.conv3_3(self.conv3_2(self.conv3_1(self.pool2(x))))
        # print("forward15")
        l2_1x = self.out2_1(x)
        # print("forward16")
        l2_2x = self.out2_2(x)
        # print("forward17")

        x = self.conv4_3(self.conv4_2(self.conv4_1(self.pool3(x))))
        l3_1x = self.out3_1(x)
        l3_2x = self.out3_2(x)
        x = self.conv5_3(self.conv5_2(self.conv5_1(self.pool4(x))))
        l4_1x = self.out4_1(x)
        l4_2x = self.out4_2(x)
        x = self.conv7(self.conv6(self.pool5(x)))
        l5_1x = self.out5_1(x)
        l5_2x = self.out5_2(x)
        # print("forward3")

        upsample1_1 = nn.functional.upsample(l5_1x + l4_1x, scale_factor=2, mode="bilinear", align_corners=True)
        upsample2_1 = nn.functional.upsample(upsample1_1 + l3_1x, scale_factor=2, mode="bilinear", align_corners=True)
        upsample3_1 = nn.functional.upsample(upsample2_1 + l2_1x, scale_factor=2, mode="bilinear", align_corners=True)
        out_1 = upsample3_1 + l1_1x
        out_1 = self.final_1(out_1)
        # print("forward4")

        upsample1_2 = nn.functional.upsample(l5_2x + l4_2x, scale_factor=2, mode="bilinear", align_corners=True)
        upsample2_2 = nn.functional.upsample(upsample1_2 + l3_2x, scale_factor=2, mode="bilinear", align_corners=True)
        upsample3_2 = nn.functional.upsample(upsample2_2 + l2_2x, scale_factor=2, mode="bilinear", align_corners=True)
        out_2 = upsample3_2 + l1_2x
        out_2 = self.final_2(out_2)
        # print("forward5")

        return [out_1, out_2]

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
