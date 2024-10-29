

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from quant_act_dorefa import ActivationQuantizer

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']
a_bit = 4


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()

        self.act_q = ActivationQuantizer(a_bits=a_bit)

        self.conv0 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.act_q(F.relu(self.bn0(self.conv0(x))))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = self.act_q(F.relu(out))
        # out = F.relu(out)
        return out

class ResNet_Cifar10(nn.Module):
    def __init__(self, block, num_blocks, inflation=1, num_classes=10):
        super(ResNet_Cifar10, self).__init__()
        self.in_planes = 32*inflation

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(32*inflation))
        ]))

        self.layer0 = self._make_layer(block, 32*inflation, num_blocks[0], stride=1)
        self.layer1 = self._make_layer(block, 64*inflation, num_blocks[1], stride=2)
        self.layer2 = self._make_layer(block, 128*inflation, num_blocks[2], stride=2)

        self.classifier = nn.Linear(128*inflation, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def resnet20():
    return ResNet_Cifar10(BasicBlock, [3, 3, 3])

def resnet32():
    return ResNet_Cifar10(BasicBlock, [5, 5, 5], inflation=1)

def resnet44():
    return ResNet_Cifar10(BasicBlock, [7, 7, 7])

def resnet56():
    return ResNet_Cifar10(BasicBlock, [9, 9, 9])

def resnet110():
    return ResNet_Cifar10(BasicBlock, [18, 18, 18])

def test(model):
    import numpy as np
    total_params = 0
    for x in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(x.data.cpu().numpy().shape)
    logging.info("Total number of params: {}".format(total_params))
    logging.info("Total layers: {}".format(len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, model.parameters())))))
