import csv

import numpy as np
from os import listdir
from tqdm import tqdm
from os.path import isfile, isdir, join
import os
import json
import random
import math

data_path = '/home/liao/VGGFace/VGGFace2/vggface2_mtcnn_160'
VGGFace_CASIA_overlap = '/home/liao/VGGFace/VGGFace2/vggface2_CASIA_overlap.txt'
VGGFace_CASIA_overlap_name = []
with open(VGGFace_CASIA_overlap, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        VGGFace_CASIA_overlap_name.append(row[0])

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
random.seed(3407)
folder_list.sort()
# label_dict = dict(zip(folder_list,range(0,len(folder_list))))

# 取前多少个类
total_classes = len(folder_list)
epoch_size = 1000
epoch = math.ceil(total_classes/epoch_size)
if epoch==0:
    print('epoch = 0 !!!')
    exit(99)
train_classfile_list_all = []
#根据总类别数（1000）和每个子集大小（epoch_size=100）计算循环次数。
# 遍历这些周期，为每个周期从folder_list中选取相应数量的类别，并获取每个类别下所有的图片文件名（并对每个类别的图片进行随机打乱顺序）。
for i in tqdm(range(epoch)):
    train_folder_list = folder_list[i * epoch_size: (i+1) * epoch_size]
    train_classfile_list_all_i = []
    for i, folder in enumerate(train_folder_list):
        if folder not in VGGFace_CASIA_overlap_name:
            folder_path = data_path+'/'+ folder
            image_list = [folder_path + '/' + cf for cf in listdir(folder_path) if (isfile(folder_path + '/' + cf) and cf[0] != '.')]
            random.shuffle(image_list)
            train_classfile_list_all_i.append(image_list)
    train_classfile_list_all = train_classfile_list_all_i + train_classfile_list_all


train_file_list = []
train_label_list = []
for i in tqdm(range(epoch)):
    train_file_list_i = []
    train_label_list_i = []
    label_start = i * epoch_size
    for i, classfile_list in enumerate(train_classfile_list_all[i * epoch_size: (i+1) * epoch_size]):
        train_file_list_i = train_file_list_i + classfile_list
        train_label_list_i = train_label_list_i + np.repeat(label_start + i, len(classfile_list)).tolist()
    train_file_list = train_file_list + train_file_list_i
    train_label_list = train_label_list + train_label_list_i
# 写入"label_names"键值，其中包含train_folder_list中的类别名称，用逗号分隔。
# 写入"image_names"键值，其中包含train_file_list中的图像文件路径，用逗号分隔。
# 写入"image_labels"键值，其中包含train_label_list中的图像标签，用逗号分隔。
train_folder_list = folder_list[:total_classes]
fo = open("vggface2_train.json", "w")
fo.write('{"label_names": [')
fo.writelines(['"%s",' % item  for item in train_folder_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write('],')

fo.write('"image_names": [')
fo.writelines(['"%s",' % item  for item in train_file_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write('],')

fo.write('"image_labels": [')
fo.writelines(['%d,' % item  for item in train_label_list])
fo.seek(0, os.SEEK_END)
fo.seek(fo.tell()-1, os.SEEK_SET)
fo.write(']}')

fo.close()
print("vggface2 train -OK")


# test_folder_list = folder_list[-500:]
# test_classfile_list_all = []
# for i, folder in enumerate(test_folder_list):
#     folder_path = data_path+'/'+ folder
#     test_classfile_list_all.append( [ folder_path+'/'+ cf for cf in listdir(folder_path) if (isfile(folder_path+'/'+cf) and cf[0] != '.')])
#     random.shuffle(test_classfile_list_all[i])
#
# test_file_list = []
# test_label_list = []
#
# for i, classfile_list in enumerate(test_classfile_list_all):
#     test_file_list = test_file_list + classfile_list
#     test_label_list = test_label_list + np.repeat(i, len(classfile_list)).tolist()
#
# fo = open("vggface2_test.json", "w")
# fo.write('{"label_names": [')
# fo.writelines(['"%s",' % item  for item in test_folder_list])
# fo.seek(0, os.SEEK_END)
# fo.seek(fo.tell()-1, os.SEEK_SET)
# fo.write('],')
#
# fo.write('"image_names": [')
# fo.writelines(['"%s",' % item  for item in test_file_list])
# fo.seek(0, os.SEEK_END)
# fo.seek(fo.tell()-1, os.SEEK_SET)
# fo.write('],')
#
# fo.write('"image_labels": [')
# fo.writelines(['%d,' % item  for item in test_label_list])
# fo.seek(0, os.SEEK_END)
# fo.seek(fo.tell()-1, os.SEEK_SET)
# fo.write(']}')
#
# fo.close()
# print("vggface2 test -OK")