import random

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import cv2
from random import sample, shuffle
from utils import load_meta, cvtColor, normalize_img
from config import HP


class YoloDataset(Dataset):

    def __init__(self, metadata_path, train=True):
        '''
        :param metadata_path: 经过预处理后的数据，为txt文件，内容每行为一条数据。每行信息包含图片路径和标签信息，它们由空格隔开。格式如下：
            /123/456/000032.jpg 104,78,375,183,0 133,88,197,123,0 195,180,213,229,14 26,189,44,238,14
        '''

        self.dataset = load_meta(metadata_path)
        self.length = len(self.dataset)
        self.isTraining = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        if self.isTraining and HP.mosaic and self.rand() < HP.mosaic_prob:
            img, box = self.parse_metadata_with_mosaic(index)
            if HP.mixup and self.rand() < HP.mixup_prob:
                img_2, box_2 = self.parse_metadata(random.randint(0, self.length - 1), random=self.isTraining)
                img, box = self.mixup_data(img, box, img_2, box_2)
        else:
            img, box = self.parse_metadata(index, random=self.isTraining)

        img_data = np.transpose(normalize_img(np.array(img, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        h, w = HP.input_shape
        box_length = len(box)
        labels_out = np.zeros((box_length, 6))
        if box_length > 0:
            box[:, [0, 2]] = box[:, [0, 2]] / w
            box[:, [1, 3]] = box[:, [1, 3]] / h

            # 将box中每项的0、1位转化为中心归一化坐标，2、3位转化为归一化后的宽高
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            # 这里出于box长度不一致的考虑，设计了一个labels_out，后处理中将在第0位填入其在当前batch中的位置信息
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return img_data, labels_out

    def rand(self, bottom=0., top=1.):
        '''
        套用一下numpy的随机值函数，简单实现一个一定范围内的随机值的功能

        :param bottom: 最小值
        :param top: 最大值
        :return: 随机数
        '''
        return np.random.rand() * (top - bottom) + bottom

    def parse_metadata(self, index, random=True, jitter=0.3, hue=0.1, sat=0.7, val=0.4):
        '''
        对单位数据进行处理，以获得输入网络的图片数据和用于计算Loss的标注数据

        :param index: 当前操作的数据的下标
        :param random: 是否进行随即增强
        :param jitter: 随机缩放的最大比例
        :param hue、sat、val: 色域变换的参数
        :return: img_data, box
        '''

        data = self.dataset[index].split()
        img_path, labels = data[0], data[1:]

        # 读取到图像，并确保其为RGB格式的数据
        img = Image.open(img_path)
        img = cvtColor(img)

        # 得到图片原始高宽（上）和输入网络的图片高宽（下）
        iw, ih = img.size
        h, w = HP.input_shape

        # 得到最原始的预测框数据
        box = np.array([np.array(list(map(int, box.split(',')))) for box in labels])

        # ---------------------------------------------------#
        # 不进行随机增强的话，本函数的所有处理将在此if语句中结束
        # ---------------------------------------------------#
        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 给图像的边缘上加上灰条，以填充满预期尺寸的图片
            img = img.resize((nw, nh), Image.BICUBIC)
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            img_data = np.array(new_img.__array__(), np.uint8)

            # 对预测框的数据进行相应的调整
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 剔除无效数据
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return img_data, box

        # ---------------------------------------------------#
        # 以下为在进行数据增强的情况下本函数的处理
        # ---------------------------------------------------#

        # 图片缩放、长宽扭曲
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2.)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        img = img.resize((nw, nh), Image.BICUBIC)

        # 给图像的边缘上加上灰条，以填充满预期尺寸的图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_img.paste(img, (dx, dy))
        img = new_img

        # 翻转图像
        flip = self.rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_data = np.array(img.__array__(), np.uint8)

        # 即将进行色域变换，计算下色域变换的参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像的色域转换到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV))
        dtype = img_data.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_HSV2RGB)

        # 对预测框的数据进行相应的调整
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:  # 如果图片经过了翻转，数据也要相应变换
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # 剔除无效数据
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return img_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        '''
        对框框的一些后处理，结合parse_metadata_with_mosaic函数处理内容一目了然
        '''
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def parse_metadata_with_mosaic(self, index, jitter=0.3, hue=0.1, sat=0.7, val=0.4):
        '''
        对单位数据进行处理，并进行了mosaic操作

        :param index: 当前操作的数据的下标
        :param jitter: 随机缩放的最大比例
        :param hue、sat、val: 色域变换的参数
        :return: img_data, box
        '''

        # 随机取出三条数据，和当前数据混在一起
        datas = sample(self.dataset, 3)
        datas.append(self.dataset[index])
        shuffle(datas)

        # 初始化一些参数，准备好摆放数据的位置
        h, w = HP.input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)
        img_datas = []
        box_datas = []

        for i, data in enumerate(datas):
            data = data.split()
            img_path, labels = data[0], data[1:]

            # 读取到图像，并确保其为RGB格式的数据
            img = Image.open(img_path)
            img = cvtColor(img)
            iw, ih = img.size

            # 得到最原始的预测框数据
            box = np.array([np.array(list(map(int, box.split(',')))) for box in labels])

            # 翻转图像
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 图片缩放、长宽扭曲
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(0.4, 1.)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            img = img.resize((nw, nh), Image.BICUBIC)

            # 根据此时图片在数据列表中的位置，确定图片位置
            global dx, dy
            if i == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif i == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif i == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif i == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            # 给图像的边缘上加上灰条，以填充满预期尺寸的图片
            new_img = Image.new('RGB', (w, h), (128, 128, 128))
            new_img.paste(img, (dx, dy))
            img_data = np.array(new_img.__array__(), np.uint8)

            # 对预测框的数据进行相应的调整
            # box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 剔除无效数据
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                # box_data = np.zeros((len(box), 5))
                # box_data[:len(box)] = box

            img_datas.append(img_data)
            # box_datas.append(box_data)
            box_datas.append(box)

        # 将图片分割，放在一起
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_img = np.zeros([h, w, 3])
        new_img[:cuty, :cutx, :] = img_datas[0][:cuty, :cutx, :]
        new_img[cuty:, :cutx, :] = img_datas[1][cuty:, :cutx, :]
        new_img[cuty:, cutx:, :] = img_datas[2][cuty:, cutx:, :]
        new_img[:cuty, cutx:, :] = img_datas[3][:cuty, cutx:, :]
        img_data = np.array(new_img, np.uint8)

        # 即将进行色域变换，计算下色域变换的参数
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像的色域转换到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(img_data, cv2.COLOR_RGB2HSV))
        dtype = img_data.dtype
        # 应用变换
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_HSV2RGB)

        # 把框调整、整合一下
        box = self.merge_bboxes(box_datas, cutx, cuty)

        return img_data, box

    def mixup_data(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = YoloDataset('testset.txt')

    data_loader = DataLoader(dataset, batch_size=2, collate_fn=yolo_dataset_collate)

    for batch in data_loader:
        img_data, box = batch
        print(img_data.size())
        print(box)
        break

    # 测试mixup
    # img, box = dataset.parse_metadata_with_mosaic(11)
    # img_2, box_2 = dataset.parse_metadata(random.randint(1, 10))
    # img, box = dataset.mixup_data(img, box, img_2, box_2)
    # img_data = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img_data = np.array(img_data, np.uint8)
    #
    # for b in box:
    #     print(b)
    #     cv2.rectangle(img_data, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness=1)
    #
    # cv2.namedWindow('123')
    # cv2.imshow('123', img_data)
    # cv2.waitKey(0)

    # 测试mosaic
    # for i in range(10):
    #     img_data, box = dataset.parse_metadata_with_mosaic(i)
    #     print(img_data.shape)
    #     img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    #
    #     for b in box:
    #         print(b)
    #         cv2.rectangle(img_data, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255), thickness=1)
    #
    #     cv2.namedWindow('123')
    #     cv2.imshow('123', img_data)
    #     cv2.waitKey(0)
