# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-04-06 10:34
# @Author : L_PenguinQAQ
# @File : plot
# @Software: PyCharm
# @function: 使用plt绘制所需要图像


import os
import matplotlib.pyplot as plt
import numpy as np


def plot_diy(data_x, func, labels, xlabel='x', ylabel='y', mode='b-', loc='best', show=True, save=''):
    data_y = func(data_x)
    p_1 = plt.plot(data_x, data_y, mode)
    # 绘制图例
    plt.legend(labels=labels, loc=loc)
    # 绘制标签
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()

    if save:
        dir = r'./plt_images'
        if not os.path.exists(dir):
            os.mkdir(dir)
        plt.savefig(f'{dir}/{save}.jpg')


if __name__ == "__main__":
    # data_x = np.linspace(0.01, 10, 100)
    # plot_diy(data_x, np.log2, ['y=-log(${{\hat{p}}_{ic}}$)'], '${{\hat{p}}_{ic}}$', 'Loss', save='Cross-Entropy Loss Function')

    data_x = np.linspace(-10, 10, 100)
    plot_diy(data_x, np.square, ['y=${\Delta^2}$'], '${\Delta=y_i-\hat{y}_i}$', 'Loss')
