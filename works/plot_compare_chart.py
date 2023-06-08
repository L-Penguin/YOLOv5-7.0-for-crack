# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-04-01 08:13
# @Author : L_PenguinQAQ
# @File : plot_diff
# @Software: PyCharm
# @function: 绘制相关数据的折线对比图

import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def csv2dic(csv):
    dic = {}
    for key in csv:
        new_key = key.strip().replace(',', '')
        dic[new_key] = csv[key]

    return dic


def curve_line(
        root,
        metric,
        files,
        names=None,
        ext=r'./results.csv',
        title='title',
        colors=None,
        lines=None
):
    if lines is None:
        lines = ['-', '-', '-', '-', '-']
    if colors is None:
        colors = ['r', 'g', 'b', 'black', 'y']
    if not names:
        names = files

    for i, name in enumerate(files):
        path = os.path.join(root, name, ext)

        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} is not exist!')

        csv = pd.read_csv(path)
        dic = csv2dic(csv)

        data_plot = list(dic[metric])
        data_plot.pop()
        x = range(len(data_plot))
        y = [d for d in data_plot]
        y_smooth = gaussian_filter1d(y, sigma=3)

        plt.plot(x, y_smooth, color=colors[i], label=names[i], linestyle=lines[i])
        plt.legend()
        plt.title(f'{title}')

    plt.show()


if __name__ == '__main__':
    curve_line(
        r'../obj-detection/train-obj',
        # 'train/box_loss',
        'metrics/mAP_0.5',
        # ['origin', 'train_1', 'train_2', 'train_3', 'train_4']
        ['模型2', '模型3']
    )

'''
    # target = 'metrics/mAP_0.5:0.95'
    target = 'test/loss'
    # target = 'test/loss'
    # names = ['模型1', '模型2', '模型3']
    names = ['efficientnet_b0--2', 'resnet34', 'yolov5s-cls']
    # root = r'../obj-detection/train-obj'
    root = r'../classify/train-cls'
    ext = r'./results.csv'

    colors = ['r', 'g', 'b']
    lines = ['-', '-', '-']
    for i, name in enumerate(names):
        path = os.path.join(root, name, ext)

        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} is not exist!')

        csv = pd.read_csv(path)
        dic = csv2dic(csv)

        data_plot = list(dic[target])
        data_plot.pop()
        x = range(len(data_plot))
        y = [d for d in data_plot]
        # y_smooth = gaussian_filter1d(y, sigma=5)
        y_smooth = gaussian_filter1d(y, sigma=3)
        # y_smooth = y

        p_1 = plt.plot(x, y_smooth, color=colors[i], label=name.split('--')[0], linestyle=lines[i])
        plt.legend()
        # plt.title(target)
        plt.title('metrics/$mAP_{50-95}$')

    plt.show()
'''
