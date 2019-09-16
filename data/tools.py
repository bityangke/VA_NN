# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 15:01
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : tools.py
# @Software: PyCharm
import numpy as np
import random


def random_choose(data, size, auto_pad=True):
    C, T, V, M = data.shape
    if T == size:
        return data
    elif T < size:
        if auto_pad:
            return auto_padding(data, size, random_pad=True)
        else:
            return data
    else:
        begin = random.randint(0, T - size)
        return data[:, begin:begin + size, :, :]


def auto_padding(data, size, random_pad=False):
    C, T, V, M = data.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_padded = np.zeros((C, size, V, M))
        data_padded[:, begin:begin + T, :, :] = data
        return data_padded
    return data


def random_rotate(data, rand_rotate):
    C, T, V, M = data.shape
    R = np.eye(3)
    for i in range(3):
        theta = (np.random.rand() * 2 - 1) * rand_rotate * np.pi
        Ri = np.zeros(3, 3)
        Ri[i, i] = 1
        Ri[(i + 1) % 3, (i + 1) % 3] = np.cos(theta)
        Ri[(i + 2) % 3, (i + 2) % 3] = np.cos(theta)
        Ri[(i + 1) % 3, (i + 2) % 3] = np.sin(theta)
        Ri[(i + 2) % 3, (i + 1) % 3] = -np.sin(theta)
        R = np.matmul(R, Ri)
    data = np.matmul(R, data.reshape(C, T * V * M)).reshape(C, T, V, M).astype('float32')
    return data
