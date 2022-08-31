import torch
from torch.utils.data import DataLoader
from model import YoloNet
from config import HP

model = YoloNet()
checkpoint = torch.load('model_save/model_4_10000.model', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])


