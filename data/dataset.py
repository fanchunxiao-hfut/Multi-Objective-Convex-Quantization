import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from .dataset_imagenet import ImageNetDownSample

# def get_dataset(config, distributed=False):
#   kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}

#   image_augmented_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#   ])

#   image_transforms = transforms.Compose([
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#   ])

#   if config.dataset == 'CIFAR10':
#     train_dataset = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=image_augmented_transforms)
#     test_dataset = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=image_transforms)

#   elif config.dataset == 'IMAGENET32':
#     # from .dataset_imagenet import ImageNetDownSample
#     train_dataset = ImageNetDownSample(root=config.data_dir, train=True, transform=image_transforms)
#     test_dataset = ImageNetDownSample(root=config.data_dir, train=False, transform=image_transforms)


#   if distributed:  
#     train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size, rank=config.local_rank, shuffle=True, drop_last=False)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, **kwargs)
#   else:
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
#   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

#   return train_loader, test_loader

def dataset_test_load(args):
    if args.dataset_test.lower()=='mnist':
        train_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_path, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_path, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize((32,32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_class = 10

    elif args.dataset_test.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_path, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_path, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_class = 10
            
    elif args.dataset_test.lower()=='cifar100':
        train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR100(args.data_path, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR100(args.data_path, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=4)
        num_class = 100
        
    return train_loader, test_loader, num_class

def dataset_load(args, order_data = True):        
    if args.data_type.lower()=='imagenet32':
        image_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        train_dataset = ImageNetDownSample(root=args.data_path, train=True, transform=image_transforms)
        test_dataset = ImageNetDownSample(root=args.data_path, train=False, transform=image_transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=order_data, num_workers=4)    #加载整体数据集时按顺序加载
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size, shuffle=False, num_workers=4)
        num_class = 1000
   
    return train_loader, test_loader, num_class

def dataset_select_load(args, indices):                                                                                          #根据索引选择固定数量的数据
    image_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    data_train_transform = ImageNetDownSample(root=args.data_path, train=True, transform=image_transforms)
    data_train_select = torch.utils.data.Subset(data_train_transform, indices)
    train_loader = torch.utils.data.DataLoader(data_train_select, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader