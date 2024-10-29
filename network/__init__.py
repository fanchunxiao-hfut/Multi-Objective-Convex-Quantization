from . import resnet, resnet_cifar10, resnet_cifar100

__all__ = ['lenet5', 'resnet18', 'resnet50', 'resnet20_cifar10', 'resnet20_cifar100', 'vgg7']
           

def resnet18():
    return resnet.resnet18()

def resnet50():
    return resnet.ResNet50()

def resnet20_cifar10():
    return resnet_cifar10.resnet20()

def resnet20_cifar100():
    return resnet_cifar100.resnet20()
