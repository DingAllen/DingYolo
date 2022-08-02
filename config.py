# 配置项目所需参数
class HyperParameters:
    # 路径配置


    # 数据配置
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7

    # 模型配置
    input_shape = (640, 640) # (height, width)

    # 训练配置
    device = 'cpu'




HP = HyperParameters()