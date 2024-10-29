# convert model and test
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import argparse
import network
from dataloader import custom_dataloader
import util.Helper as Helper
from util.LogPro import print_log_every_epoch_for_cifar, LogPro
from util.CalculateTool import AverageMeter, calculate_topK_accuracy

NUM_EPOCH = 1
GPU_ID = 0
best_prec1 = 0
best_prec5 = 0
PRINT_INTERVAL = 50


def get_params():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--pretrainedModel', type=str, default='./save_model_r20_c10_4bit/resnet20_cifar10_preqc_0.9366_342.pth',
                        help='type of dataset')
    parser.add_argument('--model',type=str,default='resnet20_cifar10',help='the choice of model',
                        choices=['lenet5', 'resnet18', 'resnet50', 'resnet20_cifar10', 'resnet20_cifar100'])
    parser.add_argument('--q_bit', type=int, default=4,help='quantize weight bit')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--end_epoch', default=3, type=int, help='number of training epoch to run')
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 128), this is the total'
                                                                    'batch size of all GPUs on the current node when '
                                                                    'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--data_path', type=str, default='./dataset', help='download dataset path')
    parser.add_argument('--data_type', type=str, default='cifar10', help='type of dataset')
    parser.add_argument('--saveckp_freq', default=299, type=int,
                        help='Save checkpoint every x epochs. Last model saving set to 299')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--workers', default=32, type=int, help='number of workers for dataloader')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args

def convert_quant(args):                                                      
    model = network.__dict__[args.model]()
    QM=-0.1                                                            
    QN=0.1
    scale = (QN-QM)/(2**args.q_bit-1)   
    print(scale)
    model_state_dict = torch.load(args.pretrainedModel, map_location='cuda:0')
    model.load_state_dict(model_state_dict)
    model.eval()


    for name, p in model.named_parameters():
        if 'conv' in name and 'bias' not in name and 'layer' in name:
            p.data = torch.round((p.data / scale))             #symmetric quantization Fq(x)
            p.data = torch.clamp(p.data, -2**(args.q_bit-1), 2**(args.q_bit-1)-1)
            p.data = p.data.to(torch.float32) * scale


    qmodel_name = "./quantized_model/resnet20_cifar10_q4_quant"
    torch.save(model.state_dict(), qmodel_name)
    print("save quantization params sucessful")
    return qmodel_name

def start_to_train(criterion, val_loader, qmodel_name, args):
    global best_prec1
    global best_prec5
    best_epoch = 1
    model = network.__dict__[args.model]().cuda()
    state_dict = torch.load(qmodel_name, map_location='cuda:0')
    model.load_state_dict(state_dict)

    Helper.calculate_model_parm_nums(model)
    for epoch in range(1, NUM_EPOCH+1):

        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

        if best_prec1 < val_prec1:
            best_prec1 = val_prec1
            best_prec5 = val_prec5
            best_epoch = epoch
        print(('prec1: {} prec5: {}').format( best_prec1 , best_prec5))


def validate(val_loader, model, criterion):
    best_prec1 = 0
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

            if batch_idx % PRINT_INTERVAL == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batch_idx, len(val_loader), loss=losses, top1=top1, top5=top5))
        return losses.val, top1.avg, top5.avg

if __name__ == '__main__':
    args = get_params()
    Helper.set_random_seed(3407)
    q_name = convert_quant(args)
    # q_name = "./quantized_model/resnet20_cifar10_q2_acc_0.9148.pth"
    train_loader, val_loader, train_sampler = custom_dataloader.dataloader(args)
    criterion = nn.CrossEntropyLoss().cuda()
    start_to_train(criterion, val_loader, q_name, args)         # test quantized model
