# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 15:08
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : feeder.py
# @Software: PyCharm
import cv2
import pickle
import numpy as np
import torch
import torch.utils.data

try:
    from data import tools
except:
    import tools


class Feeder(torch.utils.data.Dataset):
    ''' Feeder for skeleton-based action recognition
    Argument:
        data_path: the path to '.npy' data
        label_path: the path to '.pkl' label
        normalization: map skeleton sequence to image
        resize: resize image to 224x224
        random_choose: If true, randomly choose a portion of the input sequence
        random_rotate: If more than 0, randomly rotate theta angel
        window_size: The length of the output sequence
        debug: If true, only use the first 100 samples
        mmap: If true, store data in memory
    '''

    def __init__(self,
                 data_path,
                 label_path,
                 normalization=False,
                 resize=False,
                 random_choose=False,
                 random_rotate=0,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.normalization = normalization
        self.resize = resize
        self.random_choose = random_choose
        self.random_rotate = random_rotate
        self.window_size = window_size
        self.debug = debug
        self.mmap = mmap

        self.load_data()

    def load_data(self):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # load data
        if self.mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.sample_name = self.sample_name[0:100]
            self.label = self.label[0:100]
            self.data = self.data[0:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = np.array(self.data[index])
        label = np.array(self.label[index])
        if self.normalization:
            # min_map, max_map = np.array([-3.602826, -2.716611, 0.]), np.array([3.635367, 1.888282, 5.209939])
            min_map = np.array([-3.602826, -3.602826, -3.602826])
            data = np.floor(255 * (data - min_map[:, None, None, None]) / 8.812765)
        if self.resize:
            data_copy=data
            data=np.zeros((3,224,224,data.shape[-1]))
            for idx in range(data.shape[-1]):
                data[:,:,:,idx] = cv2.resize(data_copy[:,:,:,idx].transpose(2,1,0), (224, 224)).transpose(2,1,0)
        if self.random_choose:
            data = tools.random_choose(data, self.window_size)
        elif self.window_size > 0:
            data = tools.auto_padding(data, self.window_size)
        if self.random_rotate > 0:
            data = tools.random_rotate(data, self.random_rotate)
        return data, label


def fetch_dataloader(mode, params):
    if 'CV' in params['dataset_name']:
        params['train_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cv/train_data.npy'
        params['train_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cv/train_label.pkl'
        params['val_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cv/val_data.npy'
        params['val_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cv/val_label.pkl'
        params['test_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cv/test_data.npy'
        params['test_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cv/test_label.pkl'
    if 'CS' in params['dataset_name']:
        params['train_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cs/train_data.npy'
        params['train_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cs/train_label.pkl'
        params['val_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cs/val_data.npy'
        params['val_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cs/val_label.pkl'
        params['test_feeder_args']['data_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cs/test_data.npy'
        params['test_feeder_args']['label_path'] = params['dataset_dir'] + 'NTU-RGB-D' + '/cs/test_label.pkl'
    if mode == 'train':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['train_feeder_args']),
            batch_size=params['batch_size'],
            shuffle=True,
            pin_memory=False)
    if mode == 'val':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['val_feeder_args']),
            batch_size=params['batch_size'],
            shuffle=False,
            pin_memory=False)
    if mode == 'test':
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params['test_feeder_args']),
            batch_size=params['batch_size'],
            shuffle=False,
            pin_memory=False)

    return loader


if __name__ == '__main__':
    data_path = '/home/hjm/PycharmProjects/VA_NN/data/NTU-RGB-D/cv/test_data.npy'
    label_path = '/home/hjm/PycharmProjects/VA_NN/data/NTU-RGB-D/cv/test_label.pkl'
    dataset = Feeder(data_path,
                     label_path,
                     normalization=True,
                     resize=True,
                     random_choose=False,
                     random_rotate=0,
                     window_size=-1,
                     debug=False,
                     mmap=True)
    print(np.bincount(dataset.label))
