# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-04-02 14:29
# @Author : L_PenguinQAQ
# @File : test
# @Software: PyCharm
# @function: 该文件功能介绍

import cv2
import numpy as np




if __name__ == '__main__':
    img = cv2.imread('./1.jpg')
    img_1 = np.rot90(img, -1)
    cv2.imshow('1', img_1)
    cv2.waitKey(0)