import pickle
import random
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as dset
import numpy as np
from dataset import make_dataset
import random
import os
import argparse
from networks import make_model
import tf_recorder as tensorboard
from tqdm import tqdm
from utils import init_weights, calc_accuracy

parser = argparse.ArgumentParser("train")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_epoch', type=int, default=120)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=int, default=1e-4)
parser.add_argument('--save_dir', type=str, default='model')
parser.add_argument('--dataset', type=str, default='data/sort-of-clevr.pickle')
parser.add_argument('--init', type=str, default='kaiming')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--n_res', type=int, default=6)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--n_cpu', type=int, default=4)
config, _ = parser.parse_known_args()
                        
def train():
    net.train()
    losses = []
    accuracies = []

    for image, question, answer in tqdm(train_dataloader, desc='train'):
        image, question, answer = image.cuda(), question.cuda(), answer.cuda()
        pred, _ = net(image, question)
        loss = criterion(pred, answer)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tb.add_scalar('train_loss', loss)
        tb.iter()
        
        losses.append(loss.item())
        accuracy = calc_accuracy(pred, answer)
        accuracies += [accuracy] * answer.size(0)
    
    return {
        'loss': sum(losses) / len(losses),
        'acc': sum(accuracies) / len(accuracies),
    }
    
def val_(dataloader):
    net.eval()
    losses = []
    accuracies = []
    
    with torch.no_grad():
        for image, question, answer in tqdm(dataloader, desc='val'):
            image, question, answer = image.cuda(), question.cuda(), answer.cuda()
            pred, _ = net(image, question)
            loss = criterion(pred, answer)        
            losses.append(loss.item())
            accuracy = calc_accuracy(pred, answer)
            accuracies += [accuracy] * answer.size(0)
    
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)

def val():
    rel_loss, rel_acc = val_(val_rel_dataloader)
    nonrel_loss, nonrel_acc = val_(val_nonrel_dataloader)
    return {
        'rel_loss': rel_loss,
        'rel_acc': rel_acc,
        'nonrel_loss': nonrel_loss,
        'nonrel_acc': nonrel_acc,
    }

if __name__ == '__main__':
    model_dict = {
        'n_res_blocks': config.n_res,
        'n_classes': 10,
        'n_channels': 128,
    }
    os.system('mkdir ' + config.save_dir)

    np_dataset = pickle.load(open(config.dataset, 'rb'))
    np_train_dataset, np_val_dataset = np_dataset

    torch.manual_seed(config.seed)

    train_rel_dataset, train_nonrel_dataset = make_dataset(np_train_dataset, rel_augmentation=True)
    train_dataset = torch.utils.data.ConcatDataset([train_rel_dataset, train_nonrel_dataset])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.n_cpu)

    val_rel_dataset, val_nonrel_dataset = make_dataset(np_val_dataset)
    val_rel_dataloader = torch.utils.data.DataLoader(val_rel_dataset, batch_size=config.batch_size, pin_memory=True)
    val_nonrel_dataloader = torch.utils.data.DataLoader(val_nonrel_dataset, batch_size=config.batch_size, pin_memory=True)


    model_name = '{}_{}'.format(model_dict['n_res_blocks'], model_dict['n_channels'])
    tb = tensorboard.tf_recorder(model_name)
    net = make_model(model_dict).cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()

    if config.resume:
        print('load model from {}'.format(config.resume))
        prev = torch.load(config.resume)
        net.load_state_dict(prev['net'])
        optimizer.load_state_dict(prev['optimizer'])
    else:
        init_weights(net, config.init)
        
    for epoch in range(config.n_epoch):
        train_dict = train()
        val_dict = val()
        
        tb.add_scalar('val_loss', (val_dict['rel_loss'] + val_dict['nonrel_loss']) / 2)
        
        torch.save({
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, '{}/{}_epoch_{:03d}_{:.2f}_{:.2f}.pth'.format(config.save_dir, model_name, epoch, val_dict['rel_acc'], val_dict['nonrel_acc']))
        
        print('[epoch {}]'.format(epoch))
        print('train: {}'.format(train_dict))
        print('val: {}'.format(val_dict))