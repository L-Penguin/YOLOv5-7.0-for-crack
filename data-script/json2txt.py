# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2022/12/17 1:12
# @Author : L_PenguinQAQ
# @File : json2txt.py
# @Software: PyCharm
# @function: 将语义分割json文件转换为txt格式


import os
import cv2
import glob
import json
import numpy as np

def convert_json_label_to_yolov_seg_label():
    json_path = r"D:\Data\YOLO\dataSets\concreteCrackSet-seg\seg_json"
    json_files = glob.glob(json_path + "/*.json")
    for json_file in json_files:
        # if json_file != r"C:\Users\jianming_ge\Desktop\code\handle_dataset\water_street\223.json":
        #     continue
        print(json_file)
        f = open(json_file)
        json_info = json.load(f)
        # print(json_info.keys())
        img = cv2.imread(os.path.join(json_path, json_info["imagePath"]))
        height, width, _ = img.shape
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = json_file.replace(".json", ".txt")
        f = open(txt_file, "a")
        for point_json in json_info["shapes"]:
            txt_content = ""
            np_points = np.array(point_json["points"], np.int32)
            norm_points = np_points / np_w_h
            norm_points_list = norm_points.tolist()
            txt_content += "0 " + " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
            f.write(txt_content)


if __name__ == "__main__":
    convert_json_label_to_yolov_seg_label()
