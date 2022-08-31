# 配置项目所需参数
class HyperParameters:
    # 路径配置
    anchors_path = 'yolo_anchors.txt'
    metadata_train_path = 'testset.txt'
    metadata_eval_path = 'testset.txt'

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
    seed = 19990811
    batch_size = 8
    epochs = 50

HP = HyperParameters()
