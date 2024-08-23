# -*- coding:utf-8 -*-
# 一万年太久，只争朝夕
# Jeskaren
from yolo_Data import yoloDataset
import torchvision.transforms as transforms


device = 'cuda'
file_root = 'VOCdevkit/VOC2007/JPEGImages/'
batch_size = 64
momentum = 0.9
weight_decay = 0.0005
train_dataset = yoloDataset(img_root=file_root, list_file='Data_Set/Voc_train.txt', train=True, transform=[transforms.ToTensor()])

print(train_dataset[0])
