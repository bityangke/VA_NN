# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 15:01
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : ntu_generate_data.py
# @Software: PyCharm
import argparse
import os
import sys
import pickle
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split
from ntu_read_skeleton import read_xyz

training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
out_width = 30


def print_output(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(out_width):
        if i * 1.0 / out_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
    sys.stdout.write(']\n')


def generate_data(data_path,
                  out_path,
                  ignore_sample_path=None,
                  benchmark='cv',
                  dataset='test'):
    if ignore_sample_path != None:
        with open(ignore_sample_path, 'r') as f:
            ignore_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignore_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignore_samples:
            continue
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
        if benchmark == 'cv':
            training = (camera_id in training_cameras)
        elif benchmark == 'cs':
            training = (subject_id in training_subjects)
        else:
            raise ValueError()
        if dataset == 'train':
            training = training
        elif dataset == 'test':
            training = not training
        else:
            raise ValueError()
        if training:
            sample_name.append(filename)
            sample_label.append(action_class - 1)
    if dataset == 'train':
        sample_name, val_name, sample_label, val_label = train_test_split(sample_name, sample_label, test_size=0.05,
                                                                          random_state=10000)
        with open('{}/val_label.pkl'.format(out_path), 'wb') as f:
            pickle.dump((val_name, list(val_label)), f)
        f_data = open_memmap('{}/val_data.npy'.format(out_path),
                             dtype='float32',
                             mode='w+',
                             shape=(len(val_label), 3, max_frame, num_joint, max_body))
        for idx, s in enumerate(val_name):
            print_output(idx * 1.0 / len(val_label), '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '
                         .format(idx + 1, len(val_name), benchmark, 'val'))
            data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
            f_data[idx, :, 0:data.shape[1], :, :] = data
    with open('{}/{}_label.pkl'.format(out_path, dataset), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    f_data = open_memmap('{}/{}_data.npy'.format(out_path, dataset),
                         dtype='float32',
                         mode='w+',
                         shape=(len(sample_label), 3, max_frame, num_joint, max_body))
    for idx, s in enumerate(sample_name):
        print_output(idx * 1.0 / len(sample_label), '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '
                     .format(idx + 1, len(sample_name), benchmark, dataset))
        data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        f_data[idx, :, 0:data.shape[1], :, :] = data
    sys.stdout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data.')
    parser.add_argument('--data_path', default='data/NTU-RGB+D/nturgb+d_skeletons')
    parser.add_argument('--ignore_sample_path', default='data/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/NTU-RGB+D')
    benchmark = ['cs', 'cv']
    dataset = ['train', 'test']
    arg = parser.parse_args()
    for b in benchmark:
        for d in dataset:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            generate_data(arg.data_path,
                          out_path,
                          arg.ignore_sample_path,
                          benchmark=b,
                          dataset=d)
