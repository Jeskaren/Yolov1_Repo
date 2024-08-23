# -*- coding:utf-8 -*-
# 一万年太久，只争朝夕
# Jeskaren
import xml.etree.ElementTree as ET
import os
import random

# 数据集中的类别
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

# 训练集以及测试机txt文件名
train_set = open('Data_Set/Voc_train.txt', 'w')
test_set = open('Data_Set/Voc_test.txt', 'w')
# 对应数据集的xml文件处理
Annotations = 'VOCdevkit//VOC2007//Annotations//'
xml_root = os.listdir(Annotations)
random.shuffle(xml_root)
# 训练集数量以及列表
train_num = int(len(xml_root) * 0.7)
train_lists = xml_root[:train_num]
test_lists = xml_root[train_num:]


# 生成训练集和测试机的txt文件
def xml2txt():
    count = 0
    # 打开训练集和测试集文件用于写入
    with open('Data_Set/Voc_train.txt', 'w') as train_set, open('Data_Set/Voc_test.txt', 'w') as test_set:

        # 处理训练集
        for train_list in train_lists:
            # 图片文件名
            image_name = train_list.split('.')[0] + '.jpg'

            # 解析对应的 XML 文件
            results = xml_trans(Annotations + train_list)

            # 如果没有对象信息则跳过
            if len(results) == 0:
                print(train_list)
                continue

            # 写入图片文件名
            train_set.write(image_name)

            # 写入每个对象的信息
            for result in results:
                class_name = result['name']
                bbox = result['bbox']

                # 获取类名在 VOC_CLASSES 列表中的索引
                class_name = VOC_CLASSES.index(class_name)

                # 写入边界框坐标和类索引
                train_set.write(f' {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {class_name}')

            # 换行
            train_set.write('\n')

        # 处理测试集
        for test_list in test_lists:
            # 图片文件名
            image_name = test_list.split('.')[0] + '.jpg'

            # 解析对应的 XML 文件
            results = xml_trans(Annotations + test_list)

            # 如果没有对象信息则跳过
            if len(results) == 0:
                print(test_list)
                continue

            # 写入图片文件名
            test_set.write(image_name)

            # 写入每个对象的信息
            for result in results:
                class_name = result['name']
                bbox = result['bbox']

                # 获取类名在 VOC_CLASSES 列表中的索引
                class_name = VOC_CLASSES.index(class_name)

                # 写入边界框坐标和类索引
                test_set.write(f' {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {class_name}')

            # 换行
            test_set.write('\n')


# 解析xml文件，提取目标位置信息。
def xml_trans(filename):
    tree = ET.parse(filename)
    objects = []
    # 查找所有object元素
    for obj in tree.findall('object'):
        # 使用字典来存放便于后续使用
        obj_struct = {}
        # xml文件中的困难度标识，如若为1采取跳过
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue

        # 提取对象名称
        obj_struct['name'] = obj.find('name').text

        # 提取边界框信息
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]

        # 将对象信息添加到列表中
        objects.append(obj_struct)
        return objects



if __name__ == '__main__':
    xml2txt()