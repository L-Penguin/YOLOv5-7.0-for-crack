# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-04-06 10:34
# @Author : L_PenguinQAQ
# @File : plot
# @Software: PyCharm
# @function: 使用plt绘制所需要图像


import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_curve(data_x, func, labels, xlabel='x', ylabel='y', mode='b-', loc='best', show=True, save=''):
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


def plot_bar():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码

    # labels = ['方案1', '方案2', '方案3', '方案4', '方案5']
    # a = [96.8, 96.6, 97.4, 97.5, 98.0]
    # b = [84.5, 84.5, 85.3, 86.1, 86.3]
    # c = [94.5, 94.5, 96.8, 96.6, 96.7]
    # d = [65.3, 65.3, 66.8, 67.6, 67.8]
    # e = [98.0, 86.3, 96.7, 67.8]

    labels = ['$\mathregular{mAP_{50}^{mask}}$', '$\mathregular{mAP_{50-95}^{mask}}$']

    a = [94.5, 65.3]
    b = [94.5, 65.3]
    c = [96.8, 66.8]
    d = [96.6, 67.6]
    e = [97.1, 67.8]

    x = np.arange(len(labels))  # 标签位置
    width = 0.15  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width * 2, a, width, label='方案1')
    rects2 = ax.bar(x - width + 0.01, b, width, label='方案2')
    rects3 = ax.bar(x + 0.02, c, width, label='方案3')
    rects4 = ax.bar(x + width + 0.03, d, width, label='方案4')
    rects5 = ax.bar(x + width * 2 + 0.04, e, width, label='方案5')

    # 为y轴、标题和x轴等添加一些文本。
    # ax.set_ylabel('Y轴', fontsize=16)
    # ax.set_xlabel('X轴', fontsize=16)
    # ax.set_title('这里是标题')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()

    plt.ylim(60, 100)

    plt.show()


if __name__ == "__main__":
    # data_x = np.linspace(0.01, 10, 100)
    # plot_diy(data_x, np.log2, ['y=-log(${{\hat{p}}_{ic}}$)'], '${{\hat{p}}_{ic}}$', 'Loss', save='Cross-Entropy Loss Function')

    # data_x = np.linspace(-10, 10, 100)
    # plot_curve(data_x, np.square, ['y=${\Delta^2}$'], '${\Delta=y_i-\hat{y}_i}$', 'Loss')

    x = np.arange(0.00001, 1, 0.001)
    y = []
    for i in x:
        temp = -1 * math.log(i)
        y.append(temp)

    plt.plot(x, y, color='b', label='$y=-log(p_{ic})$')
    plt.legend()

    plt.xlabel('${p_{ic}}$')
    plt.ylabel('Loss')

    plt.show()