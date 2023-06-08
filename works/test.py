# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-04-02 14:29
# @Author : L_PenguinQAQ
# @File : test
# @Software: PyCharm
# @function: 测试功能模块

import os
import cv2
import numpy as np
import torch
from math import floor, ceil
import torch.nn.functional as F

ROOT = os.path.dirname(__file__)


def pic_concat(img_arr, border=False, b=1):
    """将子图数组合并生成图像
        Arguments:
            img_arr: 子图数列
            border: 是否开启子图拼接边框
            b: 边框粗细
    """
    if border:
        varr = [np.hstack([cv2.copyMakeBorder(im, b, b, b, b, borderType=cv2.BORDER_CONSTANT,
                                              value=(0, 0, 0)) for im in arr]) for arr in img_arr]
    else:
        varr = [np.hstack(arr) for arr in img_arr]

    concat_img = np.vstack(varr)

    return concat_img


def picture_seg(p, s=224):
    """将图片分割成w*w的若干子图
        Arguments:
            p: 输入图片路径
            s: 按照多少像素点分割子图
        Return:
            返回存放子图的数列，分割子图存储根路径
        Usage:
            from img_seg import picture_seg as ps
    """
    imgArr = []
    im = cv2.imread(p)

    h, w, _ = im.shape
    if h < s or w < s:
        t_1 = s // h
        t_2 = s // w

        t = max(t_1, t_2) + 1
        img = cv2.resize(im, dsize=None, fx=t, fy=t, interpolation=cv2.INTER_LINEAR)
    else:
        img = im.copy()

    # 添加的宽和高的边框像素值
    b_h = s - (img.shape[0] % s)
    b_w = s - (img.shape[1] % s)

    if b_h == s:
        b_h = 0

    if b_w == s:
        b_w = 0

    # img = cv2.copyMakeBorder(img, floor(b_h / 2), ceil(b_h / 2), floor(b_w / 2), ceil(b_w / 2),
    #                          borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    img = cv2.copyMakeBorder(img, floor(b_h / 2), ceil(b_h / 2), floor(b_w / 2), ceil(b_w / 2),
                             borderType=cv2.BORDER_REFLECT)
    h_img, w_img, _ = img.shape

    i_h, i_w = h_img // s, w_img // s

    p = os.path.split(p)[-1]
    dirName = "seg-" + p + f"-{s}"
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    for i in range(i_h):
        arr = []
        for j in range(i_w):
            img_temp = img[i * s:(i + 1) * s, j * s:(j + 1) * s]
            name = f'{i}-{j}'.center(9, '=') + '.jpg'
            path = os.path.join(dirName, name)
            cv2.imwrite(path, img_temp)
            print(f'seg img: {path} saved!')
            arr.append(img_temp)

        imgArr.append(arr)

    img_concat = pic_concat(imgArr, True, 2)
    name_concat = os.path.join(dirName, 'concat.jpg')
    cv2.imwrite(name_concat, img_concat)
    print(f'concat img: {name_concat} saved!')

    return imgArr, dirName


def albumentations(path, size=224, scale=(0.08, 1.0), ratio=(0.75, 1.0 / 0.75), jitter=0.4):
    """
    采集albumentations各种变换后的图像
    Args:
        path: 图像路径
        size: 图像resize大小
        scale:
        ratio:
        jitter:

    Returns:
        存放变换后图像的数组
    """
    # 图像路径
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} do not exist!')
    img = cv2.imread(path)
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    name = os.path.basename(path)

    # 存储albumentations变化图像根路径
    path_save = 'imgs_albumentations'
    path_save = os.path.join(ROOT, path_save)
    path_save = os.path.join(path_save, name)
    if not os.path.exists(path_save):
        try:
            os.mkdir(path_save)
        except:
            os.makedirs(path_save)

    import albumentations as A
    func_dic = {
        "RandomResizedCrop": A.RandomResizedCrop,
        "HorizontalFlip": A.HorizontalFlip,
        "VerticalFlip": A.VerticalFlip,
        "ColorJitter": A.ColorJitter,
        # "SmallestMaxSize": A.SmallestMaxSize,
        "RandomBrightnessContrast": A.RandomBrightnessContrast,
        "CoarseDropout": A.CoarseDropout,
        "ElasticTransform": A.ElasticTransform,
        "Perspective": A.Perspective
    }
    result = {key: None for key in func_dic.keys()}

    # 保存原始图像
    cv2.imwrite(os.path.join(path_save, name), img)

    for name_func in func_dic.keys():
        if name_func == 'RandomResizedCrop':
            T = [func_dic[name_func](height=size, width=size, scale=scale, ratio=ratio, p=1)]
        elif name_func == 'RandomBrightnessContrast':
            T = [A.RandomBrightnessContrast(brightness_limit=(0, 0.5), contrast_limit=0.3, brightness_by_max=True, p=1)]
        elif name_func == 'CoarseDropout':
            T = [A.CoarseDropout(max_holes=7, max_height=15, max_width=15, min_holes=1, min_height=None,
                                 min_width=None, fill_value=20, mask_fill_value=None, p=1)]
        elif name_func == 'ElasticTransform':
            T = [A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=3, value=None,
                                    mask_value=None, p=1)]
        elif name_func == 'Perspective':
            T = [A.Perspective(scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0,
                               fit_output=False, interpolation=1, p=1)]
        elif name_func == 'ColorJitter':
            color_jitter = (float(jitter),) * 3
            T = [A.ColorJitter(*color_jitter, 0)]
        else:
            T = [func_dic[name_func](p=1)]
        img_T = A.Compose(T)(image=img)["image"]

        name_T = os.path.join(path_save, f'img_{name_func}.jpg')
        cv2.imwrite(name_T, img_T)
        print(f'{os.path.basename(name_T)} is done!')

        # 变换后图像存储到返回字典中
        result[name_func] = img_T

    return result


def tensor_dilate(bin_img, ksize=3, erode=False):
    pad = (ksize - 1) // 2

    # 首先为原图加入 padding，防止图像尺寸缩小
    B, C, H, W = bin_img.shape
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='replicate', value=0)
    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    if erode:
        # 取每个 patch 中最小的值，i.e., 0
        res, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    else:
        # 取每个 patch 中最小的值，i.e., 0
        res, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)

    return res


if __name__ == '__main__':
    # picture_seg(f'./imgs/101.jpg')
    # albumentations(f'./imgs/00316.jpg')

    t = torch.zeros((15, 15))
    t[0:3, 0:3] = 1

    t_dilate = tensor_dilate(tensor_dilate(t[None, None], 5), 5, True)
    print(t, t_dilate, sep='\n')


    # t[0, 0, 4:7, 2:4] = 1
    # t[0, 0, 2:6, 5:8] = 1
    # t_dilate = tensor_dilate(t, 3, False)
    # t_erode = tensor_dilate(t_dilate, 3, True)
    # print('1', t)
    # print('2', t_dilate)
    # print('3', t_erode)

