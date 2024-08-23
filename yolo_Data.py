# -*- coding:utf-8 -*-
# 一万年太久，只争朝夕
# Jeskaren
import pandas as pd
import torch
import cv2
import os
import os.path
import random
import numpy as np
from torch.utils.data import Dataset


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 根据txt文件制作ground truth
CLASS_NUM = 20  # 使用其他训练集需要更改


class yoloDataset(Dataset):
    image_size = 448  # 定义输入图片的大小

    def __init__(self, img_root, list_file, train, transform):
        # 设置图片根目录路径
        self.root = img_root
        # 设置是否为训练集
        self.train = train
        # 设置数据预处理函数
        self.transform = transform

        # 初始化存储图片文件名、位置信息和类别信息的列表
        self.fnames = []
        self.boxes = []
        self.labels = []

        # 定义网格大小（YOLO模型的网格）
        self.S = 7
        # 定义每个网格的候选框个数
        self.B = 2
        # 定义类别数目（根据实际情况设置，例如20类）
        self.C = CLASS_NUM
        # 定义图像均值，用于归一化处理
        self.mean = (123, 117, 104)
        # 打开并读取txt文件，每行包含一个图像及其标注信息
        file_txt = open(list_file)
        lines = file_txt.readlines()
        # 逐行处理txt文件
        for line in lines:
            # 去除行首尾的空白字符，并按空白字符分割成列表
            splited = line.strip().split()
            # 存储图像文件名
            self.fnames.append(splited[0])
            # 计算每幅图像中的候选框数量
            num_boxes = (len(splited) - 1) // 5
            # 初始化存储每幅图像的位置信息和标签信息的列表
            box = []
            label = []
            # 提取位置信息和类别标签
            for i in range(num_boxes):
                # 获取每个候选框的坐标信息
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                # 获取每个候选框的类别标签（值域为0-19）
                c = splited[5 + 5 * i]
                # 将坐标和标签信息分别存储
                box.append([x, y, x2, y2])
                label.append(int(c))
            # 将每幅图像的位置信息和标签信息分别存储到boxes和labels列表中
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        # 统计总样本数（图像数量）
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 根据索引获取图像文件名
        fname = self.fnames[idx]
        # 读取图像文件
        img = cv2.imread(os.path.join(self.root, fname))
        # 从存储的盒子和标签中获取相应的图像数据，并拷贝一份以避免修改原始数据
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            # 对图像和边界框进行随机翻转
            img, boxes = self.random_flip(img, boxes)
            # 对图像和边界框进行随机缩放
            img, boxes = self.randomScale(img, boxes)
            # 对图像进行随机模糊
            img = self.randomBlur(img)
            # 对图像进行随机亮度调整
            img = self.RandomBrightness(img)
            # 对图像和边界框进行随机平移，并调整标签
            img, boxes, labels = self.randomShift(img, boxes, labels)

        # 获取图像的高度和宽度
        h, w, _ = img.shape
        # 对边界框坐标进行归一化处理，将坐标值缩放到[0,1]区间
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        # 将图像从BGR格式转换为RGB格式
        img = self.BGR2RGB(img)
        # 从图像中减去均值，以帮助网络更快收敛并提高性能
        img = self.subMean(img, self.mean)
        # 将图像调整为指定大小
        img = cv2.resize(img, (self.image_size, self.image_size))
        # 对边界框和标签进行编码，转换为YOLOv1模型的最终输出格式
        target = self.encoder(boxes, labels)
        # 应用所有预定义的数据增强变换
        for t in self.transform:
            img = t(img)
        # 返回处理后的图像和目标信息
        return img, target


    def encoder(self, boxes, labels):
        # 网格的数量
        grid_num = 7
        # 初始化一个 7x7x30 的全零张量
        target = torch.zeros((grid_num, grid_num, int(CLASS_NUM + 10)))
        # 计算每个网格的大小
        cell_size = 1. / grid_num  # 网格的边长
        # 计算每个边框的宽度和高度
        wh = boxes[:, 2:] - boxes[:, :2]
        # 计算每个边框的中心点坐标
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        # 遍历所有标注框
        for i in range(cxcy.size(0)):
            # 取出当前标注框的中心点坐标
            cxcy_sample = cxcy[i]
            # 计算该中心点属于哪个网格单元
            ij = (cxcy_sample / cell_size).ceil() - 1
            # 将该网格单元中的第一个边框的置信度设置为1
            target[int(ij[1]), int(ij[0]), 4] = 1
            # 将该网格单元中的第二个边框的置信度设置为1
            target[int(ij[1]), int(ij[0]), 9] = 1
            # 将该网格单元中的第一个边框的类别标签设置为1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 10] = 1
            # 计算网格单元左上角的坐标
            xy = ij * cell_size
            # 计算边框中心点相对于网格单元左上角的偏移量（归一化）
            delta_xy = (cxcy_sample - xy) / cell_size
            # 将边框的宽度和高度存储到目标张量中
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]  # 位置2到4用于存储宽高
            # 将边框中心点相对于网格单元左上角的偏移量存储到目标张量中
            target[int(ij[1]), int(ij[0]), :2] = delta_xy  # 位置0到2用于存储中心点偏移量
            # 将第二个边框的宽度和高度存储到目标张量中
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]  # 位置7到9用于存储第二个边框的宽高
            # 将第二个边框的中心点相对于网格单元左上角的偏移量存储到目标张量中
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy  # 位置5到7用于存储第二个边框的中心点偏移量

        # 返回编码后的目标张量
        return target  # 最终的目标张量尺寸为 7x7x30

    def BGR2RGB(self, img):
        # 将BGR图像转换为RGB格式
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        # 将BGR图像转换为HSV格式
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        # 将HSV图像转换为BGR格式
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        # 随机调整图像的亮度
        if random.random() < 0.5:
            # 以50%的概率进行亮度调整
            hsv = self.BGR2HSV(bgr)  # 将BGR图像转换为HSV格式
            h, s, v = cv2.split(hsv)  # 分离HSV通道
            adjust = random.choice([0.5, 1.5])  # 随机选择亮度调整因子（0.5或1.5）
            v = v * adjust  # 调整亮度通道
            v = np.clip(v, 0, 255).astype(hsv.dtype)  # 限制亮度值范围
            hsv = cv2.merge((h, s, v))  # 合并通道
            bgr = self.HSV2BGR(hsv)  # 将调整后的HSV图像转换回BGR格式
        return bgr

    def RandomSaturation(self, bgr):
        # 随机调整图像的饱和度
        if random.random() < 0.5:
            # 以50%的概率进行饱和度调整
            hsv = self.BGR2HSV(bgr)  # 将BGR图像转换为HSV格式
            h, s, v = cv2.split(hsv)  # 分离HSV通道
            adjust = random.choice([0.5, 1.5])  # 随机选择饱和度调整因子（0.5或1.5）
            s = s * adjust  # 调整饱和度通道
            s = np.clip(s, 0, 255).astype(hsv.dtype)  # 限制饱和度值范围
            hsv = cv2.merge((h, s, v))  # 合并通道
            bgr = self.HSV2BGR(hsv)  # 将调整后的HSV图像转换回BGR格式
        return bgr

    def RandomHue(self, bgr):
        # 随机调整图像的色相
        if random.random() < 0.5:
            # 以50%的概率进行色相调整
            hsv = self.BGR2HSV(bgr)  # 将BGR图像转换为HSV格式
            h, s, v = cv2.split(hsv)  # 分离HSV通道
            adjust = random.choice([0.5, 1.5])  # 随机选择色相调整因子（0.5或1.5）
            h = h * adjust  # 调整色相通道
            h = np.clip(h, 0, 255).astype(hsv.dtype)  # 限制色相值范围
            hsv = cv2.merge((h, s, v))  # 合并通道
            bgr = self.HSV2BGR(hsv)  # 将调整后的HSV图像转换回BGR格式
        return bgr

    def randomBlur(self, bgr):
        # 随机对图像应用模糊处理
        if random.random() < 0.5:
            # 以50%的概率应用模糊
            bgr = cv2.blur(bgr, (5, 5))  # 使用5x5的卷积核进行均值模糊
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 随机对图像进行平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2  # 计算每个框的中心
        if random.random() < 0.5:
            # 以50%的概率进行平移
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # 设置平移后图像的背景颜色
            shift_x = random.uniform(-width * 0.2, width * 0.2)  # 随机生成水平平移量
            shift_y = random.uniform(-height * 0.2, height * 0.2)  # 随机生成垂直平移量

            # 根据平移量对图像进行相应的平移处理
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)  # 计算框中心的平移量
            center = center + shift_xy  # 更新框的中心位置
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)  # 检查框中心是否在图像宽度范围内
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)  # 检查框中心是否在图像高度范围内
            mask = (mask1 & mask2).view(-1, 1)  # 综合检查框中心是否在图像范围内
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)  # 获取在图像范围内的框
            if len(boxes_in) == 0:
                return bgr, boxes, labels  # 如果没有框在图像范围内，则返回原图像及其框和标签
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)  # 计算框的平移量
            boxes_in = boxes_in + box_shift  # 更新框的位置
            labels_in = labels[mask.view(-1)]  # 更新框的标签
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        # 随机对图像进行缩放变换
        if random.random() < 0.5:
            # 以50%的概率进行缩放
            scale = random.uniform(0.8, 1.2)  # 随机生成缩放因子（0.8到1.2之间）
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))  # 根据缩放因子调整图像宽度
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)  # 计算框的缩放因子
            boxes = boxes * scale_tensor  # 更新框的位置
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        # 以0.5的概率决定是否进行随机裁剪
        if random.random() < 0.5:
            # 计算每个框的中心点
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape  # 获取图片的高度、宽度和通道数
            # 随机选择裁剪区域的高度和宽度
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            # 随机选择裁剪区域的左上角坐标
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            # 计算框的中心点在裁剪区域内的位置
            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)  # 中心点x坐标在裁剪区域宽度范围内
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)  # 中心点y坐标在裁剪区域高度范围内
            mask = (mask1 & mask2).view(-1, 1)  # 判断中心点是否在裁剪区域内

            # 选取在裁剪区域内的框
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels  # 如果没有框在裁剪区域内，返回原图和标签

            # 计算框相对于裁剪区域的位置
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)
            boxes_in = boxes_in - box_shift
            # 限制框的坐标在裁剪区域的范围内
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            # 获取裁剪区域内的标签
            labels_in = labels[mask.view(-1)]
            # 从原图中裁剪出指定区域
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        # 将图片的每个像素值减去均值
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        # 以0.5的概率决定是否进行左右翻转
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()  # 对图像进行左右翻转
            h, w, _ = im.shape
            # 调整框的坐标，使其适应翻转后的图像
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        # 随机调整图像的亮度
        alpha = random.random()
        if alpha > 0.3:
            # 按比例调整亮度，并添加随机的亮度变化
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)  # 保证像素值在[0, 255]范围内
        return im




"""
class ImgProcessor:
    # 颜色通道转换
    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # HSV格式转化
    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV转BGR
    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


    # 随机调整亮度
    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)  # 将BGR图像转换为HSV格式
            h, s, v = cv2.split(hsv)  # 分离HSV图像的H、S、V通道
            adjust = random.choice([0.5, 1.5])  # 随机选择亮度调整因子
            v = v * adjust  # 调整V通道的亮度
            v = np.clip(v, 0, 255).astype(hsv.dtype)  # 限制亮度值范围在0到255
            hsv = cv2.merge((h, s, v))  # 合并调整后的H、S、V通道
            bgr = self.HSV2BGR(hsv)  # 将HSV图像转换回BGR格式
        return bgr


    # 随机调整图像饱和度
    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)  # 将BGR图像转换为HSV格式
            h, s, v = cv2.split(hsv)  # 分离HSV图像的H、S、V通道
            adjust = random.choice([0.5, 1.5])  # 随机选择饱和度调整因子
            s = s * adjust  # 调整S通道的饱和度
            s = np.clip(s, 0, 255).astype(hsv.dtype)  # 限制饱和度值范围在0到255
            hsv = cv2.merge((h, s, v))  # 合并调整后的H、S、V通道
            bgr = self.HSV2BGR(hsv)  # 将HSV图像转换回BGR格式
        return bgr

    #  随机调整图像色调
    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)  # 将BGR图像转换为HSV格式
            h, s, v = cv2.split(hsv)  # 分离HSV图像的H、S、V通道
            adjust = random.choice([0.5, 1.5])  # 随机选择色调调整因子
            h = h * adjust  # 调整H通道的色调
            h = np.clip(h, 0, 255).astype(hsv.dtype)  # 限制色调值范围在0到255
            hsv = cv2.merge((h, s, v))  # 合并调整后的H、S、V通道
            bgr = self.HSV2BGR(hsv)  # 将HSV图像转换回BGR格式
        return bgr

    # 随机模糊图像
    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))  # 使用5x5的卷积核进行模糊
        return bgr

    # 随机平移图像，并调整目标框的位置
    def randomShift(self, bgr, boxes, labels):
        # 计算目标框的中心点
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shift_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shift_image[:, :, :] = (104, 117, 123)  # 用填充色填充图像
            shift_x = random.uniform(-width * 0.2, width * 0.2)  # 随机生成水平平移量
            shift_y = random.uniform(-height * 0.2, height * 0.2)  # 随机生成垂直平移量

            # 平移图像
            if shift_x >= 0 and shift_y >= 0:
                after_shift_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shift_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shift_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shift_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            # 更新目标框的位置
            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shift_image, boxes_in, labels_in
        return bgr, boxes, labels

    # 随机缩放图像的宽度，并调整目标框的位置
    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)  # 随机选择缩放因子
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))  # 根据缩放因子调整图像宽度
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor  # 根据缩放因子调整目标框的位置
            return bgr, boxes
        return bgr, boxes


    # 随机裁剪图像并调整目标框的位置
    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            # 计算目标框的中心点
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape

            # 随机生成裁剪区域的高度和宽度
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)

            # 随机生成裁剪区域的左上角坐标
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            # 计算目标框的中心点相对于裁剪区域的位置
            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            # 筛选出在裁剪区域内的目标框
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels

            # 计算目标框的偏移量
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            # 调整目标框的位置，并限制在裁剪区域内
            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            # 获取裁剪后的目标框标签
            labels_in = labels[mask.view(-1)]
            # 从图像中裁剪出对应区域
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels


    # 从图像中减去均值
    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr


    # 随机水平翻转图像及目标框
    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            # 水平翻转图像
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            # 更新目标框的位置
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes


    # 随机调整图像的亮度
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            # 随机调整亮度
            im = im * alpha + random.randrange(-delta, delta)
            # 限制图像像素值在 [0, 255] 范围内
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im
"""
