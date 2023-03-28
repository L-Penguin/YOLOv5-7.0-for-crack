# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/11/28 10:50
# @Author : L_PenguinQAQ
# @File : xml2txt.py
# @Software: PyCharm
# @function: xml文件转换为txt文件

import xml.etree.ElementTree as ET
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xmls_path', type=str, default='./',
                        help='directory stored xml files')
    opt = parser.parse_args()
    return opt


opt = parse_opt()
# 切换工作路径为xmls路径的根路径
os.chdir(os.path.join(opt.xmls_path, '..'))

xmlDir = os.path.basename(opt.xmls_path)

classes = ["crack"]  # 类别
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open(f'./{xmlDir}/{image_id}.xml', encoding='UTF-8')
    out_file = open('./labels/%s.txt' % image_id, 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


xml_path = os.path.join(CURRENT_DIR, opt.xmls_path)

if not os.path.isdir('./labels'):
    os.mkdir('./labels')

# xmls list
img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    print(f'{label_name} finished!')
    convert_annotation(label_name)

print(f'load xmls directory: {xml_path}')
