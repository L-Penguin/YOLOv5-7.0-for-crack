# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-02-05 21:16
# @Author : L_PenguinQAQ
# @File : extra_modules
# @Software: PyCharm
# @function: 自定义功能模块

import cv2
import torch
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import random


def show_img(img):
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        cv2.imshow('result', img)
    else:
        c = img.shape[0]
        for i in range(c):
            for j in range(c):
                cv2.imshow(f'result_{i}_{j}', img[i][j].numpy())
    cv2.waitKey(0)


def seg_img(img, c=1, stride=32):
    """将图像分割返回输出
    Arguments:
        img: 图像输入，图像输入固定为等宽高的图像
        c: 图像分块个数
    Return:
        返回ndarray格式的数据，shape为(c, c)
    """
    if c == 1:
        return img
    else:
        result = np.array([[None for _ in range(c)] for i in range(c)])
        _, _, w, h = img.shape
        s_w = w // c
        s_h = h // c
        for i in range(c):
            for j in range(c):
                result[i][j] = img[:, :, i*s_w:(i+1)*s_w, j*s_h:(j+1)*s_h]

        return result


def judge_seg(prediction, conf_thres=0.25):
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    xc = prediction[..., 4] > conf_thres

    return True in xc[0]


# 这里IOU的概念更像是只是考虑anchor的长宽
def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


# k-means聚类，且评价指标采用IOU
def k_means(boxes, k, dist=np.median, use_iou=True, use_pp=False):
    """
    yolo k-means methods
    Args:
        boxes: 需要聚类的bboxes,bboxes为n*2包含w，h
        k: 簇数(聚成几类)
        dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
        use_iou：是否使用IOU做为计算
        use_pp：是否使用k-means++算法
    """
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    # 在所有的bboxes中随机挑选k个作为簇的中心
    if not use_pp:
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
    # k_means++计算初始值
    else:
        clusters = calc_center(boxes, k)

    # print(clusters)
    while True:
        # 计算每个bboxes离每个簇的距离 1-IOU(bboxes, anchors)
        if use_iou:
            distances = 1 - wh_iou(boxes, clusters)
        else:
            distances = calc_distance(boxes, clusters)
        # 计算每个bboxes距离最近的簇中心
        current_nearest = np.argmin(distances, axis=1)
        # 每个簇中元素不在发生变化说明以及聚类完毕
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # 根据每个簇中的bboxes重新计算簇中心
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


# 计算单独一个点和一个中心的距离
def single_distance(center, point):
    center_x, center_y = center[0] / 2, center[1] / 2
    point_x, point_y = point[0] / 2, point[1] / 2
    return np.sqrt((center_x - point_x) ** 2 + (center_y - point_y) ** 2)


# 计算中心点和其他点直接的距离
def calc_distance(boxes, clusters):
    """
    :param obs: 所有的观测点
    :param clusters: 中心点
    :return:每个点对应中心点的距离
    """
    distances = []
    for box in boxes:
        # center_x, center_y = x/2, y/2
        distance = []
        for center in clusters:
            # center_xc, cneter_yc = xc/2, yc/2
            distance.append(single_distance(box, center))
        distances.append(distance)

    return distances


# k_means++计算中心坐标
def calc_center(boxes, k):
    box_number = boxes.shape[0]
    # 随机选取第一个中心点
    first_index = np.random.choice(box_number, size=1)
    clusters = boxes[first_index]
    # 计算每个样本距中心点的距离
    dist_note = np.zeros(box_number)
    dist_note += np.inf
    for i in range(k):
        # 如果已经找够了聚类中心，则退出
        if i + 1 == k:
            break
        # 计算当前中心点和其他点的距离
        for j in range(box_number):
            j_dist = single_distance(boxes[j], clusters[i])
            if j_dist < dist_note[j]:
                dist_note[j] = j_dist
        # 转换为概率
        dist_p = dist_note / dist_note.sum()
        # 使用赌轮盘法选择下一个点
        next_index = np.random.choice(box_number, 1, p=dist_p)
        next_center = boxes[next_index]
        clusters = np.vstack([clusters, next_center])
    return clusters


def cal_iou(bboxes, p):
    result = 0
    for b in bboxes:
        min_w = min(b[0], p[0])
        min_h = min(b[1], p[1])
        inter_area = min_h * min_w
        area = p[0] * p[1]
        iou = inter_area / area
        result += iou

    return result / len(bboxes)

if __name__ == '__main__':
    path = f'./img.png'
    img = cv2.imread(path)
    img = torch.from_numpy(img)
    imgs = seg_img(img, c=2)
    show_img(imgs)
