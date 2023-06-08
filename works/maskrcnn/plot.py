# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-05-08 17:05
# @Author : L_PenguinQAQ
# @File : plot_maskrcn
# @Software: PyCharm
# @function: 绘制训练过程中性能变化图

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(data):
    plt.plot(data, "r-")
    plt.show()


if __name__ == "__main__":
    path_det = r"./det.txt"
    path_seg = r"./seg.txt"

    with open(path_det) as f_det:
        content = f_det.readlines()
        for i, c in enumerate(content):
            arr = content[i].split("  ")
            arr[0] = arr[0].split(" ")[-1]
            content[i] = [arr[j] for j in (0, 1)]

        det = np.array(content, dtype=np.float16) * 100

    with open(path_seg) as f_seg:
        content = f_seg.readlines()
        for i, c in enumerate(content):
            arr = content[i].split("  ")
            arr[0] = arr[0].split(" ")[-1]
            content[i] = [arr[j] for j in (0, 1)]

        seg = np.array(content, dtype=np.float16) * 100

    res = det[:, 0] * 0.1 + det[:, 1] * 0.9 + seg[:, 0] * 0.1 + seg[:, 1] * 0.9

    max = res.max()
    index = np.argmax(res)
    print(f'max: {max}; index: {index}')
    print(f'bbox(mAP50, mAP50-95): {det[index, 0]}\t{det[index, 1]}')
    print(f'seg(mAP50, mAP50-95): {seg[index, 0]}\t{seg[index, 1]}')
    plot_curve(res)

