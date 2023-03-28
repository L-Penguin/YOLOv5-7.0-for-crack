# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/11/28 11:05
# @Author : L_PenguinQAQ
# @File : splitTrain&Val.py
# @Software: PyCharm
# @function: 将训练集和验证集分类

import os
import random
import argparse

parser = argparse.ArgumentParser()
# image文件保存路径
parser.add_argument('--imgs_path', default='images/', type=str, help='input images path')
# txt文件保存路径
parser.add_argument('--txt_path', default='path_txts/', type=str, help='output txt label path')

opt = parser.parse_args()

print(f'images path: {os.path.abspath(opt.imgs_path)}')

train_percent = 0.6  # 训练集所占比例
val_percent = 0.4  # 验证集所占比例
# test_persent = 0.1  # 测试集所占比例

imgsfilepath = opt.imgs_path
txtsavepath = opt.txt_path
total_imgs = os.listdir(imgsfilepath)

if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)


num = len(total_imgs)
list = list(range(num))

t_train = int(num * train_percent)  # 训练集数量
t_val = int(num * val_percent)  # 验证集数量

train = random.sample(list, t_train)
num1 = len(train)
for i in range(num1):
    list.remove(train[i])

val_test = [i for i in list if i not in train]
val = random.sample(val_test, t_val)
num2 = len(val)
for i in range(num2):
    list.remove(val[i])

file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')

for i in train:
    content = os.path.abspath(os.path.join(opt.imgs_path, total_imgs[i])) + '\n'
    file_train.write(content)

for i in val:
    content = os.path.abspath(os.path.join(opt.imgs_path, total_imgs[i])) + '\n'
    file_val.write(content)

for i in list:
    content = os.path.abspath(os.path.join(opt.imgs_path, total_imgs[i])) + '\n'
    file_test.write(content)

file_train.close()
file_val.close()
file_test.close()

print(f'test.txt train.txt val.txt saved in: {os.path.abspath(opt.txt_path)}')


