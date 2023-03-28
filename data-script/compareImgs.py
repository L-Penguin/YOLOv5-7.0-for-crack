# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/11/29 11:21
# @Author : L_PenguinQAQ
# @File : compareImgs.py
# @Software: PyCharm
# @function: 对比检测图和原图像

import os
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--imgSource', type=str, default='./', help='images source directory')
parser.add_argument('--detectImg', type=str, default='./', help='filename extension')
parser.add_argument('--txtPath', type=str, default='./imgsSearch.txt', help='imgSearch txt path')

opt = parser.parse_args()

txt = open(opt.txtPath, 'w')
fileArr = []
img_files = os.listdir(opt.detectImg)
for f in img_files:
    if os.path.splitext(f)[-1] in ('.jpg', '.png', '.jpeg'):
        img_1_path = os.path.join(opt.detectImg, f)
        img_2_path = os.path.join(opt.imgSource, f)
        img_1 = cv2.imread(img_1_path)
        img_1 = cv2.copyMakeBorder(img_1, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
        img_2 = cv2.imread(img_2_path)
        img_2 = cv2.copyMakeBorder(img_2, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        image = np.concatenate([img_1, img_2], axis=1)
        cv2.imshow(f'compare {f}', image)
        # 窗口展示位置固定
        cv2.moveWindow(f'compare {f}', 200, 200)
        code = cv2.waitKey(0)
        if code == 13:
            stem = os.path.splitext(f)[0]
            fileArr.append(stem)
            print(f'save {f}')
        try:
            # 关闭窗口，当鼠标点击❌关闭会报错cv2.error
            cv2.destroyWindow(f'compare {f}')
        except cv2.error:
            print("任意键关闭当前窗口")


txt.write(' '.join(fileArr))
txt.close()
print(f'save as {os.path.abspath(opt.txtPath)}')

