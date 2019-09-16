# -*-coding: utf-8*-
# @Time: 2019/8/26 下午7:24
# @Author: jiamingNo1
# FileName: main.py
# Software: PyCharm
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from datetime import datetime

from model.VA_CNN import VACNN
from model.VA_RNN import VARNN
from data.feeder import fetch_dataloader


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "y", "1")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='data/', help='root directory for all datasets')
parser.add_argument('--dataset_name', default='NTU-RGB-D-CV', help='dataset name')
parser.add_argument('--save_dir', default='weights/', help='root directory for saving checkpoint models')
parser.add_argument('--log_dir', default='logs/', help='root directory for train and test log')
parser.add_argument('--model_name', default='VACNN', help='model name')
parser.add_argument('--basenet', default='weights/resnet50.pth', help='pretrained base model')
parser.add_argument('--mode', default='train', help='train or test')
parser.add_argument('--cuda', default='True', type=str2bool, help='use cuda to train model')


def main():
    # params
    args = parser.parse_args()
    json_file = 'config/params.json'
    with open(json_file) as f:
        params = json.load(f)
    params['dataset_dir'] = args.dataset_dir
    params['dataset_name'] = args.dataset_name

    # Training settings
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("WARNING: It looks like you have a CUDA device,but aren't" +
                  "using CUDA. \nRun with --cuda for optimal training speed")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.model_name == 'VACNN':
        model = VACNN(base_model=models.resnet50()).to(device)
    elif args.model_name == 'VARNN':
        model = VARNN().to(device)
    else:
        raise ValueError()

    if not os.path.exists(args.save_dir + args.model_name):
        os.mkdir(args.save_dir + args.model_name)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # optimizer mode
    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    else:
        raise ValueError()

    # load checkpoint or base model
    max_epoch = 0
    for file in os.listdir(args.save_dir + args.model_name):
        if file.endswith('.pth'):
            basename = os.path.splitext(file)[0]
            exist_epoch = int(basename.split("{}_".format(args.model_name))[1])
            if exist_epoch > max_epoch:
                max_epoch = exist_epoch
    if max_epoch > 0:
        print('Resuming training, loading {}...'
              .format(args.save_dir + args.model_name + '/{}_{}.pth'.format(args.model_name, str(max_epoch))))
        checkpoint = torch.load(args.save_dir + args.model_name + '/{}_{}.pth'.format(args.model_name, str(max_epoch)))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        model.apply(weights_init)
        if 'CNN' in args.model_name:
            print('Loading base network...')
            resnet50_weight = torch.load(args.basenet)
            model_dict = model.state_dict()
            resnet50_weight = {k: v for k, v in resnet50_weight.items() if k in model_dict}
            model_dict.update(resnet50_weight)
            model.load_state_dict(model_dict)

    # data loader and learning rate strategy
    train_loader = fetch_dataloader('train', params)
    val_loader = fetch_dataloader('val', params)
    test_loader = fetch_dataloader('test', params)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=params['lr'], patience=2, cooldown=2, verbose=True)

    # tensorboard
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    # train and test
    if args.mode == 'train':
        writer = SummaryWriter('{}{}/'.format(args.log_dir, args.model_name) + TIMESTAMP)
        for epoch in range(params['start_epoch'], params['max_epoch']):
            train(writer, model, optimizer, device, train_loader, epoch)
            if (epoch + 1) % 5 == 0:
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()
                            },
                           args.save_dir + args.model_name + '/{}_{}.pth'.format(args.model_name, str(epoch + 1)))
                print("{:%Y-%m-%dT%H-%M-%S} saved model {}".format(
                    datetime.now(),
                    args.save_dir + args.model_name + '/{}_{}.pth'.format(args.model_name, str(epoch + 1))))
            current = val(model, device, val_loader, writer, epoch)
            lr_scheduler.step(current)
        print('Finished Training')
        writer.close()
    else:
        test(model, device, test_loader)


def train(writer, model, optimizer, device, train_loader, epoch):
    model.train()
    losses = 0.0
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=device, dtype=torch.float), target.to(device)
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        losses += loss.item()
        if (idx + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
        if (idx + 1) % 80 == 0:
            writer.add_scalar('Loss/train',
                              losses / 20,
                              epoch * (len(train_loader) // 4 + 1) + idx // 4 + 1)
            print("{:%Y-%m-%dT%H-%M-%S}  epoch:{}  (batch:{} loss:{:.2f}).".format(datetime.now(), epoch + 1,
                                                                                   idx // 4 + 1,
                                                                                   losses / 20))
            losses = 0.0


def val(model, device, val_loader, writer, epoch):
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device=device, dtype=torch.float), target.to(device)
            output = model(data)
            loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            writer.add_scalar('Accuracy/train',
                              100. * correct / len(val_loader.dataset),
                              epoch + 1)
        loss /= len(val_loader.dataset)
        print('(Val set)  Epoch:{}  Average loss: {:.2f}, Accuracy: {}/{} ({:.1f}%)'.
              format(epoch + 1, loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)))
    return 100.0 * correct / len(val_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device=device, dtype=torch.float), target.to(device)
            output = model(data)
            loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(test_loader.dataset)
        print('(Test set) Average loss: {:.2f}, Accuracy: {}/{} ({:.1f}%)'.
              format(loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif isinstance(m, nn.Linear):
        init.zeros_(m.weight.data)
        init.zeros_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight.data, 1)
        init.constant_(m.bias.data, 0)
        m.momentum = 0.99
        m.eps = 1e-3
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.orthogonal_(param)


if __name__ == '__main__':
    main()
