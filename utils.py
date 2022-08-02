import numpy as np


def load_meta(meta_path):
    with open(meta_path, 'r') as fr:
        return [line for line in fr.readlines()]


def cvtColor(image):
    '''
    将图像转换成RGB图像，防止灰度图在预测时报错。
    代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB

    :param image: PIL格式的图片数据
    :return: 转换后的图片数据
    '''
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def normalize_img(image):
    image /= 255.0
    return image


if __name__ == '__main__':
    from config import HP

    h, w = HP.input_size
    print(h)
    print(w)
