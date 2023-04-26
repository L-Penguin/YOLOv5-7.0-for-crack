# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-03-24 20:18
# @Author : L_PenguinQAQ
# @File : train_script
# @Software: PyCharm
# @function: yolov5训练脚本


import os
import re
import shutil
import sys
import argparse
from time import strftime, localtime

# 创建记录训练指令文件夹
ROOT = os.path.abspath(__file__)
ROOT = os.path.dirname(ROOT)


parser = argparse.ArgumentParser()
parser.add_argument('--task', default='seg', type=str, help='')
parser.add_argument('--weights', default='', type=str, help='')
parser.add_argument('--cfg', default='', type=str, help='')
parser.add_argument('--data', default='', type=str, help='')
parser.add_argument('--hyp', default='', type=str, help='')
parser.add_argument('--device', default=0, type=int, help='')
parser.add_argument('--imgsz', default=0, type=int, help='')
parser.add_argument('--bs', default=0, type=int, help='')
parser.add_argument('--name', default='', type=str, help='')
parser.add_argument('--rm', action='store_true', help='')
parser.add_argument('--cls-name', default='', help='')
parser.add_argument('--pre-process', action='store_true', help='')
parser.add_argument('--save-mosaic', action='store_true', help='')

opt = parser.parse_args()


def get_command(opt):
    task = opt.task
    # 根据任务跳转工作目录
    if task in ['cls', 'obj', 'seg']:
        workSpace = f"classify" if task == 'cls' else f"obj-detection" if task == 'obj' else f"segment"
        os.chdir(workSpace)
    else:
        # 任务不合法报错
        raise Exception(f'{task} task is not effective')

    # 默认输入参数
    paramsDic = {
        "weights": f"./weights/yolov5s-{task}.pt",
        "cfg": f"./cfgs/yolov5s-{task}.yaml" if task != 'cls' else '',
        "data": f"./crack-{task}.yaml",
        "hyp": f"../hyps/hyp.scratch.yaml",
        "epochs": 2000 if task != 'cls' else 100,
        "device": 0,
        "imgsz": 640 if task != 'cls' else 224,
        "batch-size": 12,
        "project": f"train-{task}",
        "name": "exp_multiScale" if task != 'cls' else '',
    }

    if task == 'obj' and opt.cls_name:
        paramsDic['cls-name'] = opt.cls_name

    # cls时的name
    if task == 'cls':
        paramsDic["name"] = os.path.splitext(os.path.split(paramsDic["weights"])[1])[0]
        paramsDic["data"] = "../dataSets/crackSet-cls"

    # 默认布尔参数
    boolsDic = {
        # 多尺度训练
        "multi-scale": True,
        # 使用缓存
        "cache": True,
        "no-overlap": True if task == 'seg' else False,
        # 损失函数
        "CIoU": True,
        "EIoU": False,
        "SIoU": False,
        # 是否使用聚类，默认为kmeans，norm范式
        "cluster": False,
        # keanspp聚类， 开启可忽略cluster
        "kmeanspp": False,
        # 聚类使用iou，忽略cluster
        "iou": False,
        "pre-process": False,
        # 是否保存mosaic图像
        "save-mosaic": False,
    }

    # weights自定义修改，限制在根目录weights中
    if opt.weights:
        paramsDic["weights"] = os.path.join(os.path.dirname(paramsDic["weights"]), opt.weights)
        if task == 'cls':
            paramsDic["name"] = os.path.splitext(os.path.split(paramsDic["weights"])[1])[0]

    # cfg自定义修改，限制在根目录和yolov5s前缀，同时修改name
    if opt.cfg:
        f = os.path.splitext(paramsDic["cfg"])
        paramsDic["cfg"] = f[0] + f'-{opt.cfg}'+f[1]
        # 修改name
        if task != 'cls':
            paramsDic["name"] = paramsDic["name"] + f'-{opt.cfg}'

    # hyp自定义修改，限制在hyps根目录
    if opt.hyp:
        f = os.path.splitext(paramsDic["hyp"])
        paramsDic["hyp"] = f[0] + f'-{opt.hyp}' + f[1]

    # data自定义修改
    if opt.data:
        paramsDic["data"] = opt.data

    # name自定义修改
    if opt.name:
        sep = '' if opt.name.isdigit() else '-'
        paramsDic["name"] = paramsDic["name"] + sep + opt.name

    # device自定义修改
    if opt.device:
        paramsDic["device"] = opt.device

    # imgsz自定义修改
    if opt.imgsz:
        paramsDic["imgsz"] = opt.imgsz

    # batch-size自定义修改
    if opt.bs:
        paramsDic["batch-size"] = opt.bs

    # 损失函数自定义修改
    if re.search(r'.IoU', opt.name):
        iou = re.search(r'.IoU', opt.name).group()
    else:
        iou = 'CIoU'

    if iou != 'CIoU':
        assert iou in boolsDic.keys(), f'{iou} is wrong loss type!'
        boolsDic["CIoU"] = False
        boolsDic[iou] = True

    # cluster自定义修改
    if paramsDic["name"].find('cluster') != -1:
        boolsDic["cluster"] = True

    # kmeanspp自定义修改
    if paramsDic["name"].find('kmeanspp') != -1:
        boolsDic["kmeanspp"] = True

    # iou聚类标准修改
    if paramsDic["name"].find('iou') != -1:
        boolsDic["iou"] = True

    # mosaic保存自定义修改
    boolsDic['save-mosaic'] = True if opt.save_mosaic else False

    # pre-process自定义修改
    if paramsDic["name"].find('pre_process') != -1:
        boolsDic["pre-process"] = True

    # 训练文件路径和log路径
    train = os.path.join(paramsDic["project"], paramsDic["name"])
    logPath = './logs'
    logName = paramsDic["name"] + '.log'
    if not os.path.exists(logPath):
        try:
            os.mkdir(logPath)
        except:
            os.makedirs(logPath)
        print(f'logPath: {os.path.abspath(logPath)} is not existed; Now Creating!')
    log = os.path.join(logPath, logName)

    # 查看文件是否存在
    check = ['weights', 'data', 'hyp'] if task == 'cls' else ['weights', 'cfg', 'data', 'hyp']
    for f in check:
        f = paramsDic[f]
        if not os.path.exists(f):
            raise FileNotFoundError(f'No such file: {f}!')

    if opt.rm:
        train_abs = os.path.abspath(train)
        log_abs = os.path.abspath(log)
        # 文件是否存在标识
        bool_train = True if os.path.exists(train_abs) else False
        bool_log = True if os.path.exists(log_abs) else False

        if bool_train and bool_log:
            res = input(f"Train: {train_abs};\nLog: {log_abs}\nDelete or not (y or n): ")
            if res.lower() == 'y':
                shutil.rmtree(train_abs)
                print(f'{train_abs} directory deleted!')
                os.remove(log_abs)
                print(f'{log_abs} file deleted!')
                sys.exit(0)
            else:
                sys.exit(0)
        elif bool_train or bool_log:
            res = input(f"{train_abs if bool_log else log_abs} is not existed!\nDelete "
                        f"{train_abs if bool_train else log_abs} or not (y or n): ")
            if res.lower() == 'y':
                if bool_train:
                    shutil.rmtree(train_abs)
                    print(f'{train_abs} directory deleted!')
                else:
                    os.remove(log_abs)
                    print(f'{log_abs} file deleted!')

                sys.exit(0)
            else:
                sys.exit(0)
        else:
            raise Exception(f'{train_abs} and {log_abs} are not existed!')

    # 查看训练文件和log是否已存在
    if os.path.exists(log):
        raise Exception(f'{log} has existed!')
    if os.path.exists(train):
        raise Exception(f'{train} has existed')

    # 有输入指令
    paramsCommand = ''
    for key_param in paramsDic.keys():
        if task == 'cls':
            if key_param not in ['cfg', 'hyp']:
                key = 'model' if key_param == 'weights' else key_param
                paramsCommand += f' --{key} {paramsDic[key_param]}'
            else:
                continue
        else:
            paramsCommand += f' --{key_param} {paramsDic[key_param]}'

    # 无输入指令
    boolsCommand = ''
    for key_bool in boolsDic.keys():
        if task == 'cls':
            break
        else:
            boolsCommand += f' --{key_bool}' if boolsDic[key_bool] else ''

    # 输出指令整合
    command = f'nohup python train.py{paramsCommand}{boolsCommand} > {log} 2>&1 &'
    tail_c = f'tail -f {log}'

    print(f'command: {command}')
    print(f'train: {train}\nlog: {log}')

    log_train = os.path.join(ROOT, './log-train')
    if not os.path.exists(log_train):
        try:
            os.mkdir(log_train)
        except:
            os.makedirs(log_train)

    # 记录各种任务的log记录
    file = os.path.join(log_train, f'train-{task}.txt')

    with open(file, mode='a', encoding='utf-8') as txt:
        sep = '=' * 20
        time = strftime('%Y-%m-%d %H:%M:%S', localtime())
        detail = ""
        if task in ['obj', 'seg']:
            # 是否使用聚类算法
            c = True if boolsDic["cluster"] or boolsDic["iou"] or boolsDic["kmeanspp"] else False
            # 使用的聚类算法
            c_k = "kmeans++" if boolsDic["kmeanspp"] else "kmeans"
            c_k = f"聚类算法: {c_k}"
            # 聚类标准
            c_c = "iou标准" if boolsDic["iou"] else "范式距离"
            c_c = f"聚类标准: {c_c}"
            detail = f'基础内容: \n' \
                     f'\tweights={paramsDic["weights"]}\t' \
                     f'\tcfg={paramsDic["cfg"]}\t' \
                     f'\tdata={paramsDic["data"]}\t' \
                     f'\thyp={paramsDic["hyp"]}\t' \
                     f'\tepochs={paramsDic["epochs"]}\t' \
                     f'\tdevice={paramsDic["device"]}\t' \
                     f'\timgsz={paramsDic["imgsz"]}\t' \
                     f'\tbatch-size={paramsDic["batch-size"]}\n' \
                     f'损失函数: {iou}\n' \
                     f'是否使用聚类算法: {"是" if c else "否"}\t{c_k if c else ""}\t{c_c if c else ""}\n' \
                     f'是否使用训练集预处理: {boolsDic["pre-process"]}'

        content = f'{sep}\n' \
                  f'{time}\n' \
                  f'运行指令：{command}\n' \
                  f'{detail}\n' \
                  f'train路径: {os.path.abspath(train)}\n' \
                  f'log路径: {os.path.abspath(log)}\n' \
                  f'{sep}\n'
        txt.write(content)
        print(f'training detail saved: {file}')

    return command, tail_c


if __name__ == "__main__":
    command, tail_c = get_command(opt)
    os.system(command)
    os.system(tail_c)
    print(ROOT)
