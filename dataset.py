from torch.utils.data.dataset import Dataset
import numpy as np
from utils import load_meta, cvtColor
from config import HP
from PIL import Image
import cv2


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
            pass
        pass

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
            if flip: # 如果图片经过了翻转，数据也要相应变换
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # 剔除无效数据
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return img_data, box

    def parse_metadata_with_mosaic(self, index, jitter=0.3, hue=0.1, sat=0.7, val=0.4):
        pass

if __name__ == '__main__':
    dataset = YoloDataset('testset.txt')

    img_data, box = dataset.parse_metadata(0, random=True)
    print(img_data)
    print(box)
