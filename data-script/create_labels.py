# D:\Anaconda3\python.exe
# -*- coding = utf-8 -*-
# @Time : 2023-02-20 09:57
# @Author : L_PenguinQAQ
# @File : create_labels
# @Software: PyCharm
# @function: 根据图像创建二分类标签


import os

if __name__ == '__main__':
    judge = False
    method = 'Positive' if judge else 'Negative'
    path = f'./concreteCrackSet_A/images/{method}'
    path_txt = path.replace(f'images/{method}', 'labels')
    for i in os.listdir(path):
        fileName = os.path.splitext(i)[0] + '.txt'
        filePath = os.path.join(path_txt, fileName)
        with open(filePath, 'w') as f:
            content = '1' if judge else '0'
            f.write(content)
            print(f'{fileName} has finished!')