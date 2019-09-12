# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 10:40
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : VA_CNN.py
# @Software: PyCharm
import math
import torch
import torch.nn as nn
import torchvision.models as models


class VACNN(nn.Module):
    '''
    Input shape should be (N,C,T,V,M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints,
          M is the number of person.
    '''

    def __init__(self,
                 base_model=None,
                 in_channel=3,
                 out_channel=128,
                 height_frame=224,
                 width_joint=224,
                 num_person=2,
                 sub_class=6,
                 num_class=60
                 ):
        super(VACNN, self).__init__()
        self.num_person = num_person
        self.sub_class = sub_class
        self.num_class = num_class
        self.sub_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.sub_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.sub_fc = nn.Linear((height_frame // 8) * (width_joint // 8) * out_channel, sub_class)
        self.resnet_layer = nn.Sequential(*list(base_model.children())[:-2])
        self.Liner_layer = nn.Linear(2048 * (height_frame // 32) * (width_joint // 32), num_class)

    def forward(self, x, target=None):
        N, C, T, V, M = x.size()
        min_map = torch.tensor([-3.602826, -3.602826, -3.602826])
        logits = []
        for idx in range(self.num_person):
            out = self.sub_conv1(x[:, :, :, :, idx])
            out = self.sub_conv2(out)
            out = out.view(out.size(0), -1)
            out = self.sub_fc(out)
            sub_out = []
            for n in range(N):
                rotation_x = torch.tensor(
                    [[1, 0, 0],
                     [0, math.cos(out[n, 0].item() * math.pi), math.sin(out[n, 0].item() * math.pi)],
                     [0, math.sin(-out[n, 0].item() * math.pi), math.cos(out[n, 0].item() * math.pi)]]
                )
                rotation_y = torch.tensor(
                    [[math.cos(out[n, 1].item() * math.pi), 0, math.sin(-out[n, 1].item() * math.pi)],
                     [0, 1, 0],
                     [math.sin(out[n, 1].item() * math.pi), 0, math.cos(out[n, 1].item() * math.pi)]]
                )
                rotation_z = torch.tensor(
                    [[math.cos(out[n, 2].item() * math.pi), math.sin(out[n, 2].item() * math.pi), 0],
                     [math.sin(-out[n, 2].item() * math.pi), math.cos(out[n, 2].item() * math.pi), 0],
                     [0, 0, 1]]
                )
                rotation = torch.mm(torch.mm(rotation_x, rotation_y), rotation_z)
                out_ = torch.mm(rotation, x[n, :, :, :, idx].view(C, T * V)) + 255 * (
                        torch.mm(rotation, (min_map[:, None] - out[n, 3:].view(-1, 1).expand(-1, T * V))) - min_map[
                                                                                                            :,
                                                                                                            None]) / 8.812765
                out_ = out_.contiguous().view(C, T, V)
                sub_out.append(out_)
            sub_out = torch.stack(sub_out).contiguous().view(N, C, T, V)
            out = self.resnet_layer(sub_out)
            logits.append(out)
        # max out logits
        out = torch.max(logits[0], logits[1])
        out = out.view(out.size(0), -1)
        out = self.Liner_layer(out)
        t = out
        assert not ((t != t).any())  # find out nan in tensor
        return out


if __name__ == '__main__':
    resnet = models.resnet50(pretrained=False)
    model = VACNN(base_model=resnet)
    children = list(model.children())
    print(children)
