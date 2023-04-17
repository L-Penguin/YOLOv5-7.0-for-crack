# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-02-05 21:16
# @Author : L_PenguinQAQ
# @File : extra_modules
# @Software: PyCharm
# @function: 自定义功能模块


import os
import sys

ROOT = os.path.dirname(__file__)
ROOT = os.path.join(ROOT, "../")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT_CLS = os.path.join(ROOT, "./classify")


from pathlib import Path
import cv2
import torch
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import random
from utils.torch_utils import select_device
from models.temp import DetectMultiBackend
from utils.augmentations import classify_transforms
import torch.nn.functional as F


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
    print(f'use_iou: {use_iou}; use_pp: {use_pp}')
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

    print(f'\ninital clusters: {clusters}\n')

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


def cal_lbp_feature(img, LBP_POINTS, LBP_RADIUS):
    h, w = img.shape
    # 构建 LBP 采样点的位置矩阵
    sample_points = np.zeros([LBP_POINTS, 2], dtype=np.int_)
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


def mkDir(path):
    if os.path.isdir(path):
        os.mkdir(path)
    else:
        raise Exception(f'{path} is not a directory')


class image_classify:
    w = os.path.join(ROOT_CLS, "./weights/yolov5s-cls.pt")
    d = os.path.join(ROOT_CLS, "./crack-cls.yaml")

    def __init__(self, weights=w, data=d, device='cpu', half=False, dnn=False):
        device = select_device(device)
        self.weights = weights
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.transforms = classify_transforms(256)

    def __call__(self, img):
        im = self.transforms(img)
        im = torch.Tensor(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        results = self.model(im)
        pred = F.softmax(results, dim=1)

        return bool(pred[0].argmax())


if __name__ == '__main__':
    weights = os.path.join(ROOT_CLS, "./train-cls/yolov5s-cls/weights/best.pt")
    img_path = os.path.join(ROOT_CLS, "./imgs-predict-cls/000.jpg")
    img = cv2.imread(img_path)
    img_cls = image_classify(weights=weights)
    p = img_cls(img)
    print(p)
