# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-04-01 08:13
# @Author : L_PenguinQAQ
# @File : plot_diff
# @Software: PyCharm
# @function: 绘制相关数据的折线对比图

import pandas as pd
import matplotlib.pyplot as plt


def csv2dic(csv):
    dic = {}
    for key in csv:
        new_key = key.strip().replace(',', '')
        dic[new_key] = csv[key]

    return dic


if __name__ == '__main__':
    path = f'./results.csv'
    csv = pd.read_csv(path)
    dic = csv2dic(csv)

    name = ['train/box_loss', 'train/seg_loss']
    for n in name:
        data_plot = list(dic[n])
        data_plot.pop()
        x = range(len(data_plot))
        y = [d for d in data_plot]
        p_1 = plt.plot(x, y, 'r-', label=n)
        plt.legend()
        plt.show()
