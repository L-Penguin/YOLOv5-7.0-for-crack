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

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='', type=str, help='')
parser.add_argument('--hyp', default='', type=str, help='')
parser.add_argument('--device', default=0, type=int, help='')
parser.add_argument('--bs', default=16, type=int, help='')
parser.add_argument('--name', default='', type=str, help='')
parser.add_argument('--iou', action='store_true', help='')
parser.add_argument('--rm', action='store_true', help='')
parser.add_argument('--save-mosaic', action='store_true', help='')

opt = parser.parse_args()

if __name__ == "__main__":
    weights = f"../weights/yolov5s-seg.pt"
    cfg = f"./cfg/yolov5s-seg{'_' if opt.cfg else ''}{opt.cfg}.yaml"
    data = f"./data/concreteCrack-seg_new.yaml"
    hyp = f"./hyp/hyp.scratch{'-low' if not opt.hyp else '_'+opt.hyp}.yaml"
    epochs = 2000
    device = opt.device
    bs = opt.bs
    cc = '--iou ' if opt.iou else ''
    if re.search(r'.IoU', opt.name):
        iou = re.search(r'.IoU', opt.name).group()
    else:
        iou = 'CIoU'
    name = f"exp_multiScale{'_' if opt.name or opt.cfg else ''}{opt.name if opt.name else opt.cfg}"

    save = f'--save-mosaic' if opt.save_mosaic else ''

    if name.find('concatSet') != -1:
        concat = "--concat-set "
    else:
        concat = ""

    if name.find('kmeanspp') != -1:
        kmeanspp = "--kmeanspp "
    else:
        kmeanspp = ""

    if name.find('mosaic9') != -1:
        mosaic9 = "--mosaic9 "
    else:
        mosaic9 = ""

    if name.find("rotate") != -1:
        rotate = "--rotate "
    else:
        rotate = ""

    # 检测文件存在
    for f in [weights, cfg, data, hyp]:
        if not os.path.exists(f):
            raise FileNotFoundError(f'No such file: {f}!')

    if os.path.exists(f'./train_new/Logs/{name}.log') and not opt.rm:
        raise Exception(f'{name}.log has existed!')

    if os.path.exists(f'./train_new/{name}') and not opt.rm:
        raise Exception(f'{name} has existed!')

    if opt.rm:
        logPath = os.path.abspath(f'./train_new/Logs/{name}.log')
        weightPath = os.path.abspath(f'./train_new/{name}')
        if not os.path.exists(logPath):
            raise FileNotFoundError(f'No such file {logPath}')
        if not os.path.exists(weightPath):
            raise FileNotFoundError(f'No such directory {weightPath}')

        res = input(f"Weights: {weightPath};\nLog: {logPath}\nDelete or not (y or n): ")
        if res.lower() == 'y':
            shutil.rmtree(weightPath)
            print(f'{name} directory deleted!')
            os.remove(logPath)
            print(f'{name}.log file deleted!')
            sys.exit(0)
        else:
            sys.exit(0)

    command = f'nohup python -u train.py --weights ../weights/yolov5s-seg.pt --cfg {cfg} ' \
              f'--data {data} --hyp {hyp} --epochs 2000 --device {device} ' \
              f'--batch-size {bs} --project train_new --name {name} --multi-scale --cache {cc}--no-overlap --{iou} ' \
              f'{concat}{kmeanspp}{rotate}{mosaic9}{save}> ./train_new/Logs/{name}.log 2>&1 &'
    print(f"{'='*25}\nname: {name}\n{'='*25}")
    print(f'device: {device};\tbatch-size: {bs}')
    if kmeanspp or cc:
        print(f'Clustering: {"kmeans++" if kmeanspp else "kmeans"};\ttype: {"iou" if cc else "norm"}')
    print(f'Using {"mosaic" + ("9" if mosaic9 else "4")} concat-set: {bool(concat)};\trotate: {bool(rotate)};\t'
          f'save-mosaic: {bool(save)}')
    print(f'Loss function: {iou}')
    a = os.system(command)
    os.system(f"tail -f ./train_new/Logs/{name}.log")


