# main document for training quantization model
import argparse
import logging
import math
import os
import random
import sys
import time

import network
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import util.Helper as Helper
from dataloader import custom_dataloader
from loss.pskd_loss import Custom_CrossEntropy_PSKD
from loss.qc_loss import QC_Loss
from torch.utils.tensorboard import SummaryWriter
from util.CalculateTool import AverageMeter, calculate_topK_accuracy
from util.LogPro import LogPro, print_log_every_epoch_for_cifar
from utils.color import Colorer


def get_params():
    parser = argparse.ArgumentParser(description='QE_code')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--lr_decay_schedule', default=[100, 225,300], nargs='*', type=int, help='when to drop lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--end_epoch', default=350, type=int, help='number of training epoch to run')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alpha_T', default=0.8, type=float, help='distillation temperature')
    parser.add_argument('--data_type', type=str, default='cifar10',choices=['mnist', 'cifar10', 'cifar100'], help='type of dataset')
    parser.add_argument('--data_path', type=str, default='/home/lmc906/EQ_PAMI/EQ_code/EQ_code/dataset', help='download dataset path')
    parser.add_argument('--model',type=str,default='resnet20_cifar10',help='the choice of model',
                        choices=['lenet5', 'resnet18','resnet50', 'resnet20_cifar10', 'resnet20_cifar100'])
    parser.add_argument('--seed', default=3407, type=int,help='seed for initializing training. ')
    parser.add_argument('--device', type=str, default='0',help='device for training')
    parser.add_argument('--log_frequency', type=int, default=50,help='每个epoch训练时每隔几个batch打印一次训练参数')
    parser.add_argument('--PSKD', default=True, action='store_true', help='是否使用自蒸馏')
    parser.add_argument('--q_bit', type=int, default=4,help='quantize weight bit')
    parser.add_argument('--a_bit', type=int, default=4,help='quantize activation bit')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--workers', default=32, type=int, help='number of workers for dataloader')
    parser.add_argument('--save_dir', default='./save_model/r20_c10_4bit', type=str, help='the path of save_model')
    parser.add_argument('--fig_path', default='fig', type=str, help='the path of loss_fig')

    args = parser.parse_args()
    return args

C = Colorer.instance()

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr

    for milestone in args.lr_decay_schedule:
        
        lr *= args.lr_decay_rate if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

def train(args, 
          all_predictions, 
          criterion_CE,
          criterion_CE_pskd,
          model, train_loader, optimizer, alpha_t, cur_epoch, quan_parm):         #通用模型训练

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    qc_losses = AverageMeter()
    kd_losses = AverageMeter()
    ce_losses = AverageMeter()


    model.train()
    for batch_idx, (data, target, input_indices) in enumerate(train_loader):      

        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        

        if args.PSKD == True:                                                                   
            target_numpy = target.cpu().detach().numpy()
            identity_matrix = torch.eye(len(train_loader.dataset.classes))        
            targets_one_hot = identity_matrix[target_numpy]                       

            if cur_epoch == 0:
                all_predictions[input_indices] = targets_one_hot

            # create new soft-targets
            soft_targets = ((1 - alpha_t) * targets_one_hot) + (alpha_t * all_predictions[input_indices])
            soft_targets = soft_targets.cuda()  

            outputs = model(data)
            softmax_output = F.softmax(outputs, dim=1) 
            loss_kl = criterion_CE_pskd(outputs, soft_targets)    
        
        else:                                                                      #不使用蒸馏
            outputs = model(data)
            loss_ce = criterion_CE(outputs, target)

        QC_loss = QC_Loss(quan_parm[0], quan_parm[1], quan_parm[2]).cuda()
        loss_qc = 0
        loss_qc_w = 0
        
                      
        for name,parms in model.named_parameters():                          #get weight
             if 'conv' in name and 'bias' not in name and 'layer' in name:
                loss_qc_w += QC_loss(parms)
                

        loss_qc = loss_qc_w
        
        optimizer.zero_grad()                                                   
        loss_qc.backward(retain_graph=True)

        for name, parms in model.named_parameters(): 
             if 'conv' in name and 'bias' not in name and 'layer' in name:
                grad_q = torch.norm(parms.grad)
                break
        
        if args.PSKD == True: 
            optimizer.zero_grad()
            loss_kl.backward(retain_graph=True)
            for name, parms in model.named_parameters(): 
                 if 'conv' in name and 'bias' not in name and 'layer' in name:
                    grad_a = torch.norm(parms.grad)
                    break
        else:
            optimizer.zero_grad()
            loss_ce.backward(retain_graph=True)
            for name, parms in model.named_parameters(): 
                 if 'conv' in name and 'bias' not in name and 'layer' in name:
                    grad_a = torch.norm(parms.grad)
                    break
        
        ##lamba setting####

        a = 1e-5
        if (cur_epoch < 50):
            lamba = 0
        elif (cur_epoch < 300):
            mag_a = torch.round(torch.log10(grad_a))
            mag_q = torch.round(torch.log10(grad_q))
            mag_poor = mag_a - mag_q
            mag_poor = mag_poor.item()
            epoch_num = math.floor(((cur_epoch - 50) / 50) + 1)
            lamba_t1 = a * 10 ** epoch_num
            lamba_t2 = 10 ** (mag_poor)
            lamba = lamba_t1 * lamba_t2
        else:
            lamba = 0.001


            
        if args.PSKD == True: 
            loss = loss_kl + lamba * loss_qc
        else:
            loss = loss_ce + lamba * loss_qc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()
        
        prec1 = calculate_topK_accuracy(outputs.data, target, topk=(1, 5))[0]
        prec5 = calculate_topK_accuracy(outputs.data, target, topk=(1, 5))[1]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        if args.PSKD == True: 
            kd_losses.update(loss_kl.item(), data.size(0))
            qc_losses.update(loss_qc.item(), data.size(0))
        else:
            ce_losses.update(loss_ce.item(), data.size(0))
            qc_losses.update(loss_qc.item(), data.size(0))

        if args.PSKD == True: 
            all_predictions[input_indices] = softmax_output.cpu().detach()
        else:
            all_predictions = None


        if args.PSKD == True:
            if batch_idx % args.log_frequency == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Lambda:{lamba}\t'
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                    'kd_Loss {loss_kd.val:.6f} ({loss_kd.avg:.6f})\t'
                    'Loss_qc {loss_qc.val:.7f} ({loss_qc.avg:.7f})\t'
                    'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                    'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    cur_epoch, batch_idx, len(train_loader), top5=top5,
                    loss=losses, loss_kd=kd_losses, loss_qc=qc_losses, top1=top1, lamba=lamba))
                loss_writer.add_scalar('loss', losses.val, cur_epoch*len(train_loader) + batch_idx)
                loss_writer.add_scalar('loss_kd', kd_losses.val, cur_epoch*len(train_loader) + batch_idx)
                loss_writer.add_scalar('loss_qc', qc_losses.val, cur_epoch*len(train_loader) + batch_idx)
        else:
            if batch_idx % args.log_frequency == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Lambda:{lamba}\t'
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                    'ce_Loss {loss_ce.val:.6f} ({loss_ce.avg:.6f})\t'
                    'Loss_qc {loss_qc.val:.7f} ({loss_qc.avg:.7f})\t'
                    'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                    'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    cur_epoch, batch_idx, len(train_loader), top5=top5,
                    loss=losses, loss_ce=ce_losses, loss_qc=qc_losses, top1=top1, lamba=lamba))
                loss_writer.add_scalar('loss', losses.val, cur_epoch*len(train_loader) + batch_idx)
                loss_writer.add_scalar('loss_ce', ce_losses.val, cur_epoch*len(train_loader) + batch_idx)
                loss_writer.add_scalar('loss_qc', qc_losses.val, cur_epoch*len(train_loader) + batch_idx)

    return all_predictions

def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):

            targets = targets.cuda()
            inputs_var = inputs.cuda()
            targets_var = targets.cuda()

            # compute output
            outputs = model(inputs_var)
            loss = criterion(outputs, targets_var)

            outputs = outputs.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = calculate_topK_accuracy(outputs.data, targets, topk=(1, 5))[0]
            prec5 = calculate_topK_accuracy(outputs.data, targets, topk=(1, 5))[1]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # if batch_idx % args.log_frequency == 0:
        print('Test:\t'
                'Loss {loss.val:.4f}\t'
                'Prec@1 {top1.avg:.3f}\t'
                'Prec@5 {top5.avg:.3f}'.format(
            batch_idx, len(val_loader), loss=losses, top1=top1, top5=top5))
        return losses.val, top1.avg, top5.avg
            
        
                
def start_to_train(args):
    
    torch.manual_seed(args.seed)                                                 
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device                              
    # args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(args.device)
    
    quan_parm = []
    QM=-0.1                                                                        
    QN=0.1
    scale = (QN-QM)/(2**args.q_bit-1)
    quan_parm.append(scale)
    quan_parm.append(QM)
    quan_parm.append(QN)
    print('Current scale = {}'.format(quan_parm[0]))
    print('Current QM, QN = {}, {}'.format(quan_parm[1], quan_parm[2]))
    
    
    
    best_prec1 = 0
    best_prec5 = 0
    best_epoch = 0

    train_loader, val_loader, train_sampler = custom_dataloader.dataloader(args)   
    model = network.__dict__[args.model]()                                        
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_schedule, 0.1)       

    if args.PSKD == True:
        print('-----Use SKD-----')
        all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)      
        print(C.underline(C.yellow("[Info] all_predictions matrix shape {}".format(all_predictions.shape))))
    else:
        print('-----Use CE-----')
        all_predictions = None

    criterion_CE = nn.CrossEntropyLoss().cuda()
    if args.PSKD == True:
        criterion_CE_pskd = Custom_CrossEntropy_PSKD().cuda() 
    else:
        criterion_CE_pskd = None

    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch, args)
        if args.PSKD == True:
            #  Alpha_t更新
            alpha_t = args.alpha_T * ((epoch + 1) / args.end_epoch)
            alpha_t = max(0, alpha_t)
        else:
            alpha_t = -1

        all_predictions = train(args,                                           
          all_predictions, 
          criterion_CE,
          criterion_CE_pskd,
          model, train_loader, optimizer, alpha_t, epoch, quan_parm)
        
        val_loss, val_prec1, val_prec5 = validate(args, val_loader, model, criterion_CE)
        
        loss_writer.add_scalar('acc', val_prec1, epoch+1)

        if best_prec1 < val_prec1:
            best_prec1 = val_prec1
            best_prec5 = val_prec5
            best_epoch = epoch
        saved_model_name = args.model + '_preqc'                                 #save model
        Helper.save_best_prec_trained_model_params(model, saved_model_name, val_prec1, save_path=args.save_dir, epoch = epoch)
        print_log_every_epoch_for_cifar(args.end_epoch, epoch, best_epoch, val_loss,
                                        best_prec1, best_prec5, val_prec1, val_prec5)

if __name__ == '__main__':
    
    args = get_params()
    
    print(args)
    
    log_dir = os.path.join('./fig', args.fig_path)
    
    loss_writer = SummaryWriter(log_dir=log_dir)
    
    start_to_train(args)
    
    
    
    
    

