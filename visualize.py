import pickle
import torch
import numpy as np
import argparse
from dataset import make_dataset
from networks import make_model
from tqdm import tqdm
from utils import init_weights, calc_accuracy
import torch.nn.functional as F
import torchvision.utils as vutils
import os

parser = argparse.ArgumentParser("test")
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--n_res', type=int, default=6)
parser.add_argument('--save_dir', type=str, default='features')
config, _ = parser.parse_known_args()

batch_size = 1
resume = config.model

model_dict = {
    'n_res_blocks': config.n_res,
    'n_classes': 10,
    'n_channels': 128,
}

np_dataset = pickle.load(open(config.dataset, 'rb'))
_, np_val_dataset = np_dataset
val_rel_dataset, _ = make_dataset(np_val_dataset)
val_dataloader = torch.utils.data.DataLoader(val_rel_dataset, batch_size=batch_size,
                                             pin_memory=True, drop_last=False)

net = make_model(model_dict).cuda()

print('load model from {}'.format(resume))
prev = torch.load(resume)
net.load_state_dict(prev['net'])

net.eval()

try:
    os.makedirs(config.save_dir)
except:
    pass

colors = ['red', 'green', 'blue', 'orange', 'gray', 'yellow']
questions = ['shape?', 'left?', 'up?', 'closest?', 'furthest?', 'count?']
answers = ['yes', 'no', 'rectangle', 'circle', '1', '2', '3', '4', '5', '6']

for i, (image, _, _) in enumerate(val_dataloader):
    color = input('color? ')

    if color not in colors:
        print('noop')
        continue

    origin = image[:1, ...]
    image = image[:1, ...].repeat(6, 1, 1, 1)
    
    # 모든 종류의 질문 생성
    question = torch.zeros([6, 11])
    question[:, colors.index(color)] = 1.0

    question[0:3, 6] = 1.0
    question[3:6, 7] = 1.0

    question[0, 8], question[1, 9], question[2, 10] = 1.0, 1.0, 1.0
    question[3, 8], question[4, 9], question[5, 10] = 1.0, 1.0, 1.0

    image, question  = image.cuda(), question.cuda()
    with torch.no_grad():
        pred, feature = net(image, question)

    for q, p in zip(questions, pred):
        print('{} {}'.format(q, answers[torch.argmax(p)]))

    feature = torch.relu(feature)
    feature = torch.sum(feature, dim=1, keepdim=True)
    feature = feature / torch.max(feature.view(6, -1), dim=1)[0].view(6, 1, 1, 1)
    feature = F.interpolate(feature, size=image.size(2), mode='bilinear')

    origin = origin.cuda()
    origin = origin[:, (2, 1, 0), ...]
    image = image[:, (2, 1, 0), ...]
    image = image * feature
    image = torch.cat([origin, image])
    vutils.save_image(image, '{}/{:5d}.png'.format(config.save_dir, i), nrow=8)