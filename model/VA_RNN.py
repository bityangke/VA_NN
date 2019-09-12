# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 10:40
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : VA_RNN.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn


class VARNN(nn.Module):
    '''
    Input shape should be (N,C,T,V,M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints,
          M is the number of person.
    '''

    def __int__(self,
                dim_point=3,
                num_joint=25,
                num_person=2,
                num_class=60,
                sub_hidden=256,
                main_hidden=256
                ):
        super(VARNN, self).__init__()
        self.dim_point = dim_point
        self.num_joint = num_joint
        self.num_person = num_person
        self.num_class = num_class
        self.sub_rotation_lstm = nn.LSTM(dim_point * num_joint, sub_hidden)
        self.sub_rotation_fc = nn.Linear(sub_hidden, 3)
        self.sub_translation_lstm = nn.LSTM(dim_point * num_joint, sub_hidden)
        self.sub_translation_fc = nn.Linear(sub_hidden, 3)
        self.main_lstm = nn.LSTM(dim_point * num_joint, main_hidden, 3)
        self.main_fc = nn.Linear(main_hidden, num_class)

    def forward(self, x, target=None):
        N, C, T, V, M = x.size()
        logits = []
        for idx in range(self.num_person):
            sub_out = x.permute(0, 2, 1, 3, 4)[:, :, :, :, idx]
            sub_out = self.sub_rotation_lstm(sub_out.contiguous().view(N, T, -1))
            sub_rotation_out = self.sub_rotation_lstm(sub_out)
            sub_rotation_out = self.sub_rotation_fc(sub_rotation_out)
            sub_translation_out = self.sub_translation_lstm(sub_out)
            sub_translation_out = self.sub_translation_fc(sub_translation_out)
            sub_out = []
            for n in range(N):
                rotation_x = torch.tensor([[1, 0, 0],
                                           [0, np.cos(sub_rotation_out[n, 0]*np.pi), np.sin(sub_rotation_out[n, 0]*np.pi)],
                                           [0, np.sin(-sub_rotation_out[n, 0]*np.pi), np.cos(sub_rotation_out[n, 0]*np.pi)]])
                rotation_y = torch.tensor([[np.cos(sub_rotation_out[n, 1]*np.pi), 0, np.sin(-sub_rotation_out[n, 1]*np.pi)],
                                           [0, 1, 0],
                                           [np.sin(sub_rotation_out[n, 1]*np.pi), 0, np.cos(sub_rotation_out[n, 1]*np.pi)]])
                rotation_z = torch.tensor([[np.cos(sub_rotation_out[n, 2]*np.pi), np.sin(sub_rotation_out[n, 2]*np.pi), 0],
                                           [np.sin(-sub_rotation_out[n, 2]*np.pi), np.cos(sub_rotation_out[n, 2]*np.pi), 0],
                                           [0, 0, 1]])
                x_sub = x[n, :, :, :, idx] - torch.tensor(sub_translation_out[n, :]).view(1, -1, 1, 1, 1).expand(1, C, T, V, 1)
                x_sub = x_sub.permute(1, 0, 2, 3, 4).contiguous().view(C, T * V)
                out = torch.mm(torch.mm(torch.mm(rotation_x, rotation_y), rotation_z), x_sub)
                out = out.contiguous().view(C, 1, T, V, 1).permute(1, 0, 2, 3, 4)
                sub_out.append(out)
            out = self.main_lstm(torch.tensor(sub_out))
            logits.append(out)

        # max out logits
        out = torch.max(logits[0], logits[1])
        out = out.view(out.size(0), -1)
        out = self.main_fc(out)
        t = out
        assert not ((t != t).any())  # find out nan in tensor
        assert not (t.abs().sum() == 0)  # find out 0 tensor
        return out


if __name__ == '__main__':
    pass
