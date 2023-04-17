# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-03-14 16:16
# @Author : L_PenguinQAQ
# @File : work_LBP
# @Software: PyCharm
# @function: 测试LBP对混凝土裂纹图像处理
import os

import cv2
import numpy as np

# 定义 LBP 采样点和采样区域半径
LBP_POINTS = 24
LBP_RADIUS = 7


# LBP 特征提取函数
def cal_lbp_feature(img):
    h, w = img.shape
    # 构建 LBP 采样点的位置矩阵
    sample_points = np.zeros([LBP_POINTS, 2], dtype=np.int)
    for i in range(LBP_POINTS):
        sample_points[i][0] = int(round(LBP_RADIUS * np.cos(i * 2 * np.pi / LBP_POINTS)))
        sample_points[i][1] = int(round(LBP_RADIUS * np.sin(i * 2 * np.pi / LBP_POINTS)))
    # 计算 LBP 特征
    lbp_feature = np.zeros([h - 2 * LBP_RADIUS, w - 2 * LBP_RADIUS], dtype=np.uint8)
    for i in range(LBP_RADIUS, h - LBP_RADIUS):
        for j in range(LBP_RADIUS, w - LBP_RADIUS):
            center_pixel = img[i, j]
            feature_value = 0
            for k in range(LBP_POINTS):
                x = i + sample_points[k][0]
                y = j + sample_points[k][1]
                if img[x, y] > center_pixel:
                    feature_value += 2 ** k
            lbp_feature[i - LBP_RADIUS, j - LBP_RADIUS] = feature_value
    return lbp_feature


# 演示 LBP 特征提取的过程
if __name__ == '__main__':
    path = r'D:\Data\YOLO\dataSets\concreteCrackSet-LBP\images'
    for f in os.listdir(path):
        imgPath = os.path.join(path, f)
        img = cv2.imread(imgPath, 0)
        lbp_feature = cal_lbp_feature(img)
        lbp_feature = cv2.resize(lbp_feature, img.shape[0:2])
        cv2.imwrite(imgPath, lbp_feature)
        print(f'{imgPath} finished!!!')
