# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 10:44
# @Author  : jiamingNo1
# @Email   : jiaming19.huang@foxmail.com
# @File    : ntu_read_skeleton.py
# @Software: PyCharm
import os
import cv2
import numpy as np


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # (3,frame_nums,25 2)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data


def draw_skeleton(data, frame_idx, frame):
    skeleton_line = [[1, 13], [1, 17], [1, 2], [2, 21], [21, 5], [21, 9],
                     [21, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11],
                     [11, 12], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]]
    for body in range(2):
        for joint in range(25):
            x = data[0, frame_idx, joint, body]
            y = data[1, frame_idx, joint, body]
            cv2.circle(frame, (int(x * 800), int(500 - y * 500)), 5, (255, 0, 0), -1)
        for idx in range(len(skeleton_line)):
            joint1 = skeleton_line[idx][0] - 1
            joint2 = skeleton_line[idx][1] - 1
            x1 = data[0, frame_idx, joint1, body]
            y1 = data[1, frame_idx, joint1, body]
            x2 = data[0, frame_idx, joint2, body]
            y2 = data[1, frame_idx, joint2, body]
            cv2.line(frame, (int(x1 * 800), int(500 - y1 * 500)), (int(x2 * 800), int(500 - y2 * 500)), (0, 0, 255), 2)


def show_skeleton_rgb(file):
    data = read_xyz(file)
    for frame_idx in range(data.shape[1]):
        canvas = np.zeros((500, 500, 3), dtype='uint8') + 255
        draw_skeleton(data, frame_idx, canvas)
        cv2.imshow('skeleton', canvas)
        cv2.waitKey(100)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data_path = 'data/NTU-RGB+D/nturgb+d_skeletons'
    test_skeleton = 'S001C001P001R001A001.skeleton'
    data = read_xyz(os.path.join(data_path, test_skeleton))
    print(data, data.shape)

    # show_skeleton_rgb(os.path.join(data_path, test_skeleton))

    # n = 1
    # c_max = [0, 0, 0]
    # c_min = [1, 1, 1]
    # for file in os.listdir('data/NTU-RGB+D/nturgb+d_skeletons'):
    #     print(n)
    #     n += 1
    #     data = read_xyz(os.path.join('data/NTU-RGB-D/nturgb+d_skeletons', file))
    #
    #     for idx in range(3):
    #         if np.min(data[idx, :, :, :]) < c_min[idx]:
    #             c_min[idx] = np.min(data[idx, :, :, :])
    #     for idx in range(3):
    #         if np.max(data[idx, :, :, :]) > c_max[idx]:
    #             c_max[idx] = np.max(data[idx, :, :, :])
    # print(c_max)
    # print(c_min)
