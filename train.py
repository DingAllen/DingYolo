import os.path
import random
import torch
import numpy as np
from config import HP
from tensorboardX import SummaryWriter
from model import YoloNet, YoloLoss
from dataset import YoloDataset, yolo_dataset_collate
from argparse import ArgumentParser
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from utils import get_lr_scheduler, set_optimizer_lr, get_anchors, weights_init

logger = SummaryWriter('./log')

# seed init: 保证模型的可复现性
torch.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)
torch.cuda.manual_seed(HP.seed)


def evaluate(model, devloader, crit):
    model.eval()
    sum_loss = 0.
    with torch.no_grad():
        for batch in devloader:
            x, y = batch
            pred = model(x)
            loss = crit(pred, y.to(HP.device), x.to(HP.device))
            sum_loss += loss.item()

    model.train()
    return sum_loss / len(devloader)


def save_checkpoint(model, epoch, opt, save_path):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict()
    }
    torch.save(save_dict, save_path)


def train():
    parser = ArgumentParser(description="Model Training")
    parser.add_argument('--c', default=None, type=str, help='train from scratch or resume from training')
    args = parser.parse_args()

    model = YoloNet()
    weights_init(model)

    anchors, num_anchors = get_anchors(HP.anchors_path)
    criterion = YoloLoss(anchors=anchors)

    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    momentum = 0.937
    weight_decay = 5e-4

    nbs = 64
    lr_limit_max = 5e-2
    lr_limit_min = 5e-4
    Init_lr_fit = min(max(HP.batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(HP.batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    opt = optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
    opt.add_param_group({"params": pg1, "weight_decay": weight_decay})
    opt.add_param_group({"params": pg2})

    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, HP.epochs)

    trainset = YoloDataset(HP.metadata_train_path)
    train_loader = DataLoader(trainset, batch_size=HP.batch_size, shuffle=True, drop_last=True, collate_fn=yolo_dataset_collate, num_workers=4)

    devset = YoloDataset(HP.metadata_eval_path, train=False)
    dev_loader = DataLoader(devset, batch_size=HP.batch_size, shuffle=True, drop_last=False, collate_fn=yolo_dataset_collate, num_workers=4)

    start_epoch, step = 0, 0

    if args.c:
        checkpoint = torch.load(args.c)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Resume from %s.' % args.c)
    else:
        print('Training from scratch.')

    model = model.to(HP.device)
    model.train()

    for epoch in range(start_epoch, HP.epochs):
        print('Start Epoch: %d, Steps: %d' % (epoch, len(train_loader)))
        set_optimizer_lr(opt, lr_scheduler_func, epoch)
        for batch in train_loader:
            x, y = batch  # 加载数据
            opt.zero_grad()  # 梯度归零
            pred = model(x)
            loss = criterion(pred, y.to(HP.device), x.to(HP.device))

            loss.backward()
            opt.step()

            logger.add_scalar('Loss/Train', loss, step)

            if not step % HP.verbose_step:
                eval_loss = evaluate(model, dev_loader, criterion)
                logger.add_scalar('Loss/Dev', eval_loss, step)

            if not step % HP.save_step:
                model_path = 'model_%d_%d.model' % (epoch, step)
                save_checkpoint(model, epoch, opt, os.path.join('model_save', model_path))

            step += 1
            logger.flush()
            print('Epoch:[%d/%d], step:%d, Train Loss:%.5f, Dev Loss:%.5f' % (
                epoch, HP.epochs, step, loss.item(), eval_loss))

    logger.close()


if __name__ == '__main__':
    train()
