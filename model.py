import torch
from torch import nn
from config import HP
from torch.nn import functional as F
import math
import numpy as np

class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CBS, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activition = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activition(x)
        return x


class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELAN, self).__init__()

        c = in_channels // 2
        self.cbs1 = CBS(in_channels, c, 1, 1)
        self.cbs2 = CBS(in_channels, c, 1, 1)
        self.cbs3 = CBS(c, c, 3, 1)
        self.cbs4 = CBS(c, c, 3, 1)
        self.cbs5 = CBS(c, c, 3, 1)
        self.cbs6 = CBS(c, c, 3, 1)
        self.cbs_all = CBS(c * 4, out_channels, 3, 1)

    def forward(self, x):
        x1 = self.cbs1(x)
        x2 = self.cbs2(x)
        x3 = self.cbs4(self.cbs3(x2))
        x4 = self.cbs6(self.cbs5(x3))
        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        return self.cbs_all(x_all)


class ELAN_H(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELAN_H, self).__init__()

        c1 = in_channels // 2
        c2 = c1 // 2
        self.cbs1 = CBS(in_channels, c1, 1, 1)
        self.cbs2 = CBS(in_channels, c1, 1, 1)
        self.cbs3 = CBS(c1, c2, 3, 1)
        self.cbs4 = CBS(c2, c2, 3, 1)
        self.cbs5 = CBS(c2, c2, 3, 1)
        self.cbs6 = CBS(c2, c2, 3, 1)
        self.cbs_all = CBS(c1 * 4, out_channels, 3, 1)

    def forward(self, x):
        x1 = self.cbs1(x)
        x2 = self.cbs2(x)
        x3 = self.cbs3(x2)
        x4 = self.cbs4(x3)
        x5 = self.cbs5(x4)
        x6 = self.cbs6(x5)
        x_all = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        return self.cbs_all(x_all)


class MP1(nn.Module):
    def __init__(self, in_channels):
        super(MP1, self).__init__()
        c = in_channels // 2
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbs1 = CBS(in_channels, c, 1, 1)
        self.cbs2 = CBS(in_channels, c, 1, 1)
        self.cbs3 = CBS(c, c, 3, 2)

    def forward(self, x):
        x1 = self.cbs1(self.max_pool(x))
        x2 = self.cbs3(self.cbs2(x))
        return torch.cat([x1, x2], dim=1)


class MP2(nn.Module):
    def __init__(self, in_channels):
        super(MP2, self).__init__()
        c = in_channels
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbs1 = CBS(in_channels, c, 1, 1)
        self.cbs2 = CBS(in_channels, c, 1, 1)
        self.cbs3 = CBS(c, c, 3, 2)

    def forward(self, x):
        x1 = self.cbs1(self.max_pool(x))
        x2 = self.cbs3(self.cbs2(x))
        return torch.cat([x1, x2], dim=1)


class SPPCSPC(nn.Module):
    def __init__(self, in_channels, out_channels, k=(5, 9, 13)):
        # 参数k参考yolov7的官方代码
        super(SPPCSPC, self).__init__()
        c = in_channels
        self.cbs1 = CBS(in_channels, c, 1, 1)
        self.cbs2 = CBS(in_channels, c, 1, 1)
        self.cbs3 = CBS(c, c, 3, 1)
        self.cbs4 = CBS(c, c, 1, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cbs5 = CBS(4 * c, c, 1, 1)
        self.cbs6 = CBS(c, c, 3, 1)
        self.cbs7 = CBS(2 * c, out_channels, 1, 1)

    def forward(self, x):
        y1 = self.cbs1(x)

        x = self.cbs4(self.cbs3(self.cbs2(x)))
        y2 = self.cbs6(self.cbs5(torch.cat([x] + [pool(x) for pool in self.pools], dim=1)))

        return self.cbs7(torch.cat([y1, y2], dim=1))


class UP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.cbs = CBS(in_channels, out_channels, 1, 1)
        self.unsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.unsample(self.cbs(x))


class RepConv_Without_Identity(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(RepConv_Without_Identity, self).__init__()
        self.act = nn.SiLU()
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.03)
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.03)
        )

    def forward(self, x):
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x))


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.extractor = nn.Sequential(
            CBS(3, 32, 3, 1),
            CBS(32, 64, 3, 2),
            CBS(64, 64, 3, 1),
            CBS(64, 128, 3, 2),
            ELAN(128, 256)
        )
        self.calcu1 = nn.Sequential(
            MP1(256),
            ELAN(256, 512)
        )
        self.calcu2 = nn.Sequential(
            MP1(512),
            ELAN(512, 1024)
        )
        self.calcu3 = nn.Sequential(
            MP1(1024),
            ELAN(1024, 1024)
        )

    def forward(self, x):
        x = self.extractor(x)
        x3 = self.calcu1(x)
        x2 = self.calcu2(x3)
        x1 = self.calcu3(x2)
        return x1, x2, x3


class YoloNet(nn.Module):
    def __init__(self):
        super(YoloNet, self).__init__()
        self.backbone = Backbone()
        self.cbs2 = CBS(1024, 256, 1, 1)
        self.cbs3 = CBS(512, 128, 1, 1)

        self.sppcspc = SPPCSPC(1024, 512)
        self.up1 = UP(512, 256)
        self.elan1 = ELAN_H(512, 256)
        self.up2 = UP(256, 128)
        self.elan2 = ELAN_H(256, 128)

        self.mp2_1 = MP2(128)
        self.elan3 = ELAN_H(512, 256)
        self.mp2_2 = MP2(256)
        self.elan4 = ELAN_H(1024, 512)

        self.repconv1 = RepConv_Without_Identity(512, 1024)
        self.repconv2 = RepConv_Without_Identity(256, 512)
        self.repconv3 = RepConv_Without_Identity(128, 256)

        self.head1 = nn.Conv2d(1024, len(HP.anchor_mask[0]) * (5 + HP.num_classes), (1, 1))
        self.head2 = nn.Conv2d(512, len(HP.anchor_mask[1]) * (5 + HP.num_classes), (1, 1))
        self.head3 = nn.Conv2d(256, len(HP.anchor_mask[2]) * (5 + HP.num_classes), (1, 1))

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        x1 = self.sppcspc(x1)
        x2 = self.elan1(torch.cat([self.cbs2(x2), self.up1(x1)], dim=1))
        x3 = self.elan2(torch.cat([self.cbs3(x3), self.up2(x2)], dim=1))

        x2 = self.elan3(torch.cat([x2, self.mp2_1(x3)], dim=1))
        x1 = self.elan4(torch.cat([x1, self.mp2_2(x2)], dim=1))

        x1 = self.repconv1(x1)
        x2 = self.repconv2(x2)
        x3 = self.repconv3(x3)

        x1 = self.head1(x1)
        x2 = self.head2(x2)
        x3 = self.head3(x3)

        return [x1, x2, x3]


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class YoloLoss(nn.Module):
    def __init__(self, anchors):
        super(YoloLoss, self).__init__()
        self.anchors = [anchors[mask] for mask in HP.anchor_mask]
        self.balance = [0.4, 1.0, 4]
        self.stride = [32, 16, 8]

        self.box_ratio = 0.05
        self.obj_ratio = 1 * (HP.input_shape[0] * HP.input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (HP.num_classes / 80)
        self.threshold = 4

        self.cp, self.cn = smooth_BCE(eps=0)
        self.BCEcls, self.BCEobj, self.gr = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(), 1

    def __call__(self, predictions, targets, imgs):
        # 将预测结果变换成方便编程的模样
        for i in range(len(predictions)):
            n, _, h, w = predictions[i].size()
            predictions[i] = predictions[i].view(n, len(HP.anchor_mask[i]), -1, h, w).permute(0, 1, 3, 4,
                                                                                              2).contiguous()

        # 初始化三个部分的loss值
        cls_loss = torch.zeros(1, device=HP.device)
        box_loss = torch.zeros(1, device=HP.device)
        obj_loss = torch.zeros(1, device=HP.device)

        # 正样本匹配
        bs, as_, gjs, gis, targets, anchors = self.build_targets(predictions, targets, imgs)

        # 计算得到各个特征层的[w, h, w, h]，目的应该是方便下面去直接乘以预测结果以直接获得各个特征层对应的目标位置
        feature_map_sizes = [torch.tensor(prediction.shape, device=HP.device)[[3, 2, 3, 2]].type_as(prediction) for
                             prediction in predictions]

        # 对三个特征层依次进行loss的计算
        for i, prediction in enumerate(predictions):
            # -------------------------------------------#
            #   image, anchor, gridy, gridx
            # -------------------------------------------#
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # 第几张图片，第几个真实框，y，x
            tobj = torch.zeros_like(prediction[..., 0], device=HP.device)  # target obj

            # -------------------------------------------#
            #   获得目标数量，如果目标大于0
            #   则开始计算种类损失和回归损失
            # -------------------------------------------#
            n = b.shape[0]
            if n:
                prediction_pos = prediction[b, a, gj, gi]  # prediction subset corresponding to targets

                # -------------------------------------------#
                #   计算匹配上的正样本的回归损失
                # -------------------------------------------#
                # -------------------------------------------#
                #   grid 获得正样本的x、y轴坐标
                # -------------------------------------------#
                grid = torch.stack([gi, gj], dim=1)
                # -------------------------------------------#
                #   进行解码，获得预测结果
                # -------------------------------------------#
                xy = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                wh = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                box = torch.cat((xy, wh), 1)
                # -------------------------------------------#
                #   对真实框进行处理，映射到特征层上
                # -------------------------------------------#
                selected_tbox = targets[i][:, 2:6] * feature_map_sizes[i]
                selected_tbox[:, :2] -= grid.type_as(prediction)
                # -------------------------------------------#
                #   计算预测框和真实框的回归损失
                # -------------------------------------------#
                iou = self.bbox_iou(box.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                box_loss += (1.0 - iou).mean()
                # -------------------------------------------#
                #   根据预测结果的iou获得置信度损失的gt
                # -------------------------------------------#
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # -------------------------------------------#
                #   计算匹配上的正样本的分类损失
                # -------------------------------------------#
                selected_tcls = targets[i][:, 1].long()
                t = torch.full_like(prediction_pos[:, 5:], self.cn, device=HP.device)  # targets
                t[range(n), selected_tcls] = self.cp
                cls_loss += self.BCEcls(prediction_pos[:, 5:], t)  # BCE

            # -------------------------------------------#
            #   计算目标是否存在的置信度损失
            #   并且乘上每个特征层的比例
            # -------------------------------------------#
            obj_loss += self.BCEobj(prediction[..., 4], tobj) * self.balance[i]  # obj loss

            # -------------------------------------------#
            #   将各个部分的损失乘上比例
            #   全加起来后，乘上batch_size
            # -------------------------------------------#
        box_loss *= self.box_ratio
        obj_loss *= self.obj_ratio
        cls_loss *= self.cls_ratio
        bs = tobj.shape[0]

        loss = box_loss + obj_loss + cls_loss
        return loss

    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        box2 = box2.T

        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union

        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        else:
            return iou  # IoU

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    # 匹配正样本
    def build_targets(self, predictions, targets, imgs):
        # -------------------------------------------#
        #   匹配正样本
        # -------------------------------------------#

        # indices = [ 3 * (图片的index, 当前图片的第几个先验框, 当前先验框的x, 当前先验框的y) ]
        indices, anch = self.find_3_positive(predictions, targets)

        matching_bs = [[] for _ in predictions]
        matching_as = [[] for _ in predictions]
        matching_gjs = [[] for _ in predictions]
        matching_gis = [[] for _ in predictions]
        matching_targets = [[] for _ in predictions]
        matching_anchs = [[] for _ in predictions]

        # -------------------------------------------#
        #   一共三层
        # -------------------------------------------#
        num_layer = len(predictions)
        # -------------------------------------------#
        #   对batch_size进行循环，进行OTA匹配
        #   在batch_size循环中对layer进行循环
        # -------------------------------------------#
        for batch_idx in range(predictions[0].shape[0]):  # for i in range(n)
            # -------------------------------------------#
            #   先判断匹配上的真实框哪些属于该图片
            # -------------------------------------------#
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            # -------------------------------------------#
            #   如果没有真实框属于该图片则continue
            # -------------------------------------------#
            if this_target.shape[0] == 0:
                continue

            # -------------------------------------------#
            #   真实框的坐标进行缩放
            # -------------------------------------------#
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            # -------------------------------------------#
            #   从中心宽高到左上角右下角
            # -------------------------------------------#
            txyxy = self.xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            # -------------------------------------------#
            #   对三个layer进行循环
            # -------------------------------------------#
            for i, prediction in enumerate(predictions):
                # -------------------------------------------#
                #   b代表第几张图片 a代表第几个先验框
                #   gj代表y轴，gi代表x轴
                # -------------------------------------------#
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]

                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                # -------------------------------------------#
                #   取出这个真实框对应的预测结果
                # -------------------------------------------#
                fg_pred = prediction[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                # -------------------------------------------#
                #   获得网格后，进行解码
                # -------------------------------------------#
                grid = torch.stack([gi, gj], dim=1).type_as(fg_pred)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = self.xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            # -------------------------------------------#
            #   判断是否存在对应的预测框，不存在则跳过
            # -------------------------------------------#
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue

            # -------------------------------------------#
            #   进行堆叠
            # -------------------------------------------#
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            # -------------------------------------------------------------#
            #   计算当前图片中，真实框与预测框的重合程度
            #   iou的范围为0-1，取-log后为0~inf
            #   重合程度越大，取-log后越小
            #   因此，真实框与预测框重合度越大，pair_wise_iou_loss越小
            # -------------------------------------------------------------#
            pair_wise_iou = self.box_iou(txyxy, pxyxys)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            # -------------------------------------------#
            #   最多二十个预测框与真实框的重合程度
            #   然后求和，找到每个真实框对应几个预测框
            # -------------------------------------------#
            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            # -------------------------------------------#
            #   gt_cls_per_image    种类的真实信息
            # -------------------------------------------#
            gt_cls_per_image = F.one_hot(this_target[:, 1].to(torch.int64), HP.num_classes).float().unsqueeze(
                1).repeat(1, pxyxys.shape[0], 1)

            # -------------------------------------------#
            #   cls_preds_  种类置信度的预测信息
            #               cls_preds_越接近于1，y越接近于1
            #               y / (1 - y)越接近于无穷大
            #               也就是种类置信度预测的越准
            #               pair_wise_cls_loss越小
            # -------------------------------------------#
            num_gt = this_target.shape[0]
            cls_preds_ = p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * p_obj.unsqueeze(0).repeat(num_gt,
                                                                                                                1,
                                                                                                                1).sigmoid_()
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(torch.log(y / (1 - y)), gt_cls_per_image,
                                                                    reduction="none").sum(-1)
            del cls_preds_

            # -------------------------------------------#
            #   求cost的总和
            # -------------------------------------------#
            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            # -------------------------------------------#
            #   求cost最小的k个预测框
            # -------------------------------------------#
            matching_matrix = torch.zeros_like(cost)
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks

            # -------------------------------------------#
            #   如果一个预测框对应多个真实框
            #   只使用这个预测框最对应的真实框
            # -------------------------------------------#
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            # -------------------------------------------#
            #   取出符合条件的框
            # -------------------------------------------#
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
            this_target = this_target[matched_gt_inds]

            for i in range(num_layer):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(num_layer):
            matching_bs[i] = torch.cat(matching_bs[i], dim=0) if len(matching_bs[i]) != 0 else torch.Tensor(
                matching_bs[i])
            matching_as[i] = torch.cat(matching_as[i], dim=0) if len(matching_as[i]) != 0 else torch.Tensor(
                matching_as[i])
            matching_gjs[i] = torch.cat(matching_gjs[i], dim=0) if len(matching_gjs[i]) != 0 else torch.Tensor(
                matching_gjs[i])
            matching_gis[i] = torch.cat(matching_gis[i], dim=0) if len(matching_gis[i]) != 0 else torch.Tensor(
                matching_gis[i])
            matching_targets[i] = torch.cat(matching_targets[i], dim=0) if len(
                matching_targets[i]) != 0 else torch.Tensor(matching_targets[i])
            matching_anchs[i] = torch.cat(matching_anchs[i], dim=0) if len(matching_anchs[i]) != 0 else torch.Tensor(
                matching_anchs[i])

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, predictions, targets):
        num_anchor, num_gt = len(HP.anchor_mask[0]), targets.shape[0]
        indices, anchors = [], []
        gain = torch.ones(7, device=HP.device)
        # ------------------------------------#
        #   ai      [num_anchor, num_gt]
        #   targets [num_gt, 6] => [num_anchor, num_gt, 7]
        # ------------------------------------#
        ai = torch.arange(num_anchor, device=HP.device).float().view(num_anchor, 1).repeat(1, num_gt)
        targets = torch.cat((targets.repeat(num_anchor, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # offsets
        off = torch.tensor([
            [0, 0],
            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], device=targets.device).float() * g

        for i in range(len(predictions)):

            # 得到当前特征层维度上的先验框
            anchors_i = torch.from_numpy(self.anchors[i] / self.stride[i]).type_as(predictions[i])

            # gain[2:6] = [w, h, w, h]
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]

            # 将位置映射在特征层上
            t = targets * gain

            if num_gt:
                # 真实框的[宽, 高] / anchor的[宽, 高]
                r = t[:, :, 4:6] / anchors_i[:, None]
                # 通过阈值筛选所用的真实框
                j = (torch.max(r, 1. / r).max(2)[0] < self.threshold)
                t = t[j]

                # gxy为真实框映射再特征层中的位置
                gxy = t[:, 2:4]
                # gxi为gxy与特征层右下角的距离
                gxi = gain[[2, 3]] - gxy

                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))

                # -------------------------------------------#
                #   t   重复5次，使用满足条件的j进行框的提取
                #   j   一共五行，代表当前特征点在五个
                #       [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]
                #       方向是否存在
                # -------------------------------------------#

                # [5, num_matched_anchor, 7]     mask = [5, num_matched_anchor] -> [num_matched_anchor * 3, 7]
                t = t.repeat((5, 1, 1))[j]

                # [1, num_matched_anchor, 2] + [5, 1, 2] -> [5, num_matched_anchor, 2]       mask = [5, num_matched_anchor] -> [num_matched_anchor * 3, 2]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # -------------------------------------------#
            #   b   代表属于第几个图片
            #   gxy 代表该真实框所处的x、y中心坐标
            #   gwh 代表该真实框的wh坐标
            #   gij 代表真实框所属的特征点坐标
            # -------------------------------------------#
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # -------------------------------------------#
            #   gj、gi不能超出特征层范围
            #   a代表属于该特征点的第几个先验框
            # -------------------------------------------#
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anchors.append(anchors_i[a])  # anchors

        return indices, anchors


if __name__ == '__main__':
    import time

    x = torch.randn((1, 3, 640, 640))
    net = YoloNet()
    t1 = time.time()
    x1, x2, x3 = net(x)
    t2 = time.time()
    print(t2 - t1)
    print(x1.size())
    print(x2.size())
    print(x3.size())
