# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/11/29 10:58
# @Author : L_PenguinQAQ
# @File : imgsSearch.py
# @Software: PyCharm
# @function: 用于寻找检测有误的图像文件到指定文件夹内进行手动标注

import os
import shutil
import argparse


# 复制函数
def copyfile(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % srcfile)
    else:
        # 分离文件名和路径
        fpath, fname = os.path.split(srcfile)
        # 创建路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        # 复制文件
        shutil.copy(srcfile, f'{dstpath}/{fname}')
        print("copy %s -> %s" % (srcfile, f'{dstpath}/{fname}'))


parser = argparse.ArgumentParser()
parser.add_argument('--imgSource', type=str, default='./', help='images source directory')
parser.add_argument('--ext', type=str, default='jpg', help='filename extension')
parser.add_argument('--txtPath', type=str, default='./imgsSearch.txt', help='txt file loaded for searching')
parser.add_argument('--dstPath', type=str, default='./img_search', help='directory to save images')

opt = parser.parse_args()

f = open(opt.txtPath, 'r')

fileArr = ''.join(f.readlines()).replace('\n', ' ').split(' ')

dst = opt.dstPath
# 清空目的文件夹
if os.path.exists(dst):
    shutil.rmtree(dst)
    os.makedirs(dst)

# 将源文件夹内文件复制到目的文件夹
for s in fileArr:
    fileName = f'{s.zfill(5)}.{opt.ext}'
    src = os.path.join(opt.imgSource, fileName)
    copyfile(src, dst)

print(f'copy imgs number: {len(fileArr)}')

print(f'imgSource: {os.path.abspath(opt.imgSource)}')
print(f'loaded txt file: {os.path.abspath(opt.txtPath)}')
print(f'copy the {opt.ext} files to: {os.path.abspath(opt.dstPath)}')