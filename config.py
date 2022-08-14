# 配置项目所需参数
class HyperParameters:
    # 路径配置
    anchor_path = 'data/yolo_anchors.txt'

    # 数据配置
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7

    # 模型配置
    input_shape = (640, 640)  # (height, width)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 80

    # 训练配置
    device = 'cpu'


HP = HyperParameters()
