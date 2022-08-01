from torch.utils.data.dataset import Dataset
import numpy as np
from utils import load_meta, cvtColor
from config import HP
from PIL import Image


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

    def rand(self, bottom=0, top=1):
        '''
        套用一下numpy的随机值函数，简单实现一个一定范围内的随机值的功能

        :param bottom: 最小值
        :param top: 最大值
        :return: 随机数
        '''
        return np.random.rand() * (top - bottom) + bottom

    def parse_data(self, index, random=True, jitter=.3, hue=.1, sat=0.7, val=0.4):
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
            img_data = np.array(new_img, np.float32)

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
        pass


if __name__ == '__main__':
    dataset = YoloDataset('testset.txt')

    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode

    resize_img = T.Resize((608, 608), interpolation=InterpolationMode.BOX)
    img, box = dataset.parse_data(2, random=False)
    print(box)
    img.show()
