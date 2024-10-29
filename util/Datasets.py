import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super(self).__iter__())

class DataDrefetcher():
    """
    Usage:
    prefetcher = dataset.DataDrefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        forward()
        input, target = prefetcher.next()
        i += 1
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.preload()
        return self.next_input, self.next_target


def get_mnist_dataset_loader(root_path=None, batch_size=100, val_batch_size=100):
    """
    make train/validate dataset loader
    :param root_path: path/to/mnist-dataset
    :param batch_size:batch size of train dataset
    :param test_batch_size: batch size of validate dataset
    :return: train dataset loader and validate dataset loader
    """
    if root_path is None:
        raise ValueError("'root_path' parameter cannot be None")

    _mnist_stats = {
        'mean': [0.5],
        'std': [0.5],
    }

    kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root_path, train=True, download=False,
                           transform=transforms.Compose([
                               # transforms.Pad(2),
                               # transforms.RandomCrop((32, 32), padding=4),
                               transforms.ToTensor(),
                               # transforms.Normalize(**_mnist_stats)
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root_path, train=False, transform=transforms.Compose([
                               transforms.Pad(2),
                               transforms.ToTensor(),
                               # transforms.Normalize(**_mnist_stats)
                           ])),
            batch_size=val_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_cifar10_dataset_loader(root_path=None, batch_size=100, val_batch_size=100, augment=False):
    """
    make train/validate dataset loader
    :param root_path: path/to/cifar10-dataset
    :param batch_size:batch size of train dataset
    :param test_batch_size: batch size of validate dataset
    :param augment: data-augment,default is False
    :return: train dataset loader and validate dataset loader
    """
    if root_path is None:
        raise ValueError("'root_path' parameter cannot be None")

    _cifar10_stats = {
        'mean': [0.4914, 0.4822, 0.4465],
        'std' : [0.2023, 0.1994, 0.2010]
    }

    _train_transform = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**_cifar10_stats),
        ])
    _val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**_cifar10_stats),
        ])
    if augment:
        _train_transform = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**_cifar10_stats),
        ])

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(
        datasets.CIFAR10(root=root_path,
                     train=True,
                     transform=_train_transform,
                     download=True), batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = DataLoader(
        datasets.CIFAR10(root=root_path,
                     train=False,
                     transform=_val_transform,
                     download=True), batch_size=val_batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader

def get_cifar100_dataset_loader(root_path=None, batch_size=100, val_batch_size=100, augment=True):
    """
    make train/validate dataset loader
    :param root_path: path/to/cifar10-dataset
    :param batch_size:batch size of train dataset
    :param test_batch_size: batch size of validate dataset
    :param augment: data-augment,default is False
    :return: train dataset loader and validate dataset loader
    """
    if root_path is None:
        raise ValueError("'root_path' parameter cannot be None")

    _cifar100_stats = {
        'mean': [0.4914, 0.4822, 0.4465],
        'std' : [0.2023, 0.1994, 0.2010]
    }

    _transform = []


    _train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**_cifar100_stats),
        ])
    _val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**_cifar100_stats),
        ])
    if augment:
        _train_transform = transforms.Compose([
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**_cifar100_stats),
        ])

    kwargs = {'num_workers':4, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(
        datasets.CIFAR100(root=root_path,
                     train=True,
                     transform=_train_transform,
                     download=True), batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = DataLoader(
        datasets.CIFAR100(root=root_path,
                     train=False,
                     transform=_val_transform,
                     download=True), batch_size=val_batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader

def get_svhn_dataset_loader(root_path=None, batch_size=100, val_batch_size=100):
    """
    make train/validate dataset loader
    :param root_path: path/to/dataset
    :param batch_size:batch size of train dataset
    :param test_batch_size: batch size of validate dataset
    :return: train dataset loader and validate dataset loader
    """
    if root_path is None:
        raise ValueError("'root_path' parameter cannot be None")

    _svhn_stats = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    }

    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**_svhn_stats),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=root_path,
                      split='train',
                      transform=_transform,
                      download=False),
                      batch_size=batch_size, shuffle=True, **kwargs)

    extra_train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=root_path,
                      split='extra',
                      transform=_transform,
                      download=False),
                      batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=root_path,
                      split='test',
                      transform=_transform,
                      download=False), batch_size=val_batch_size, shuffle=False, **kwargs)

    return train_loader, extra_train_loader, val_loader

def get_imagenet_dataset_loader(root_path=None, batch_size=100, val_batch_size=100, augment=True):
    """
    make train/validate dataset loader
    :param root_path: path/to/imagenet-dataset
    :param batch_size:batch size of train dataset
    :param test_batch_size: batch size of validate dataset
    :return: train dataset loader and validate dataset loader
    """
    if root_path is None:
        raise ValueError("'root_path' parameter cannot be None")

    _imagenet_stats = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }

    _transform = []
    if augment:
        _train_transform = transforms.Compose([
                                transforms.Resize((256,256)),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(**_imagenet_stats)
        ])
        _val_transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(**_imagenet_stats)
        ])

    else:
        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**_imagenet_stats),
        ])

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
                    root=os.path.join(root_path, 'train'),
                    transform=_train_transform),
                    batch_size=batch_size, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
                    root=os.path.join(root_path, 'val'),
                    transform=_val_transform),
                    batch_size=val_batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader