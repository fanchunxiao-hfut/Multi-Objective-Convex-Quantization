#-*- coding: utf-8 -*-
import torch
import time
import numpy as np
import random
import torch.nn as nn
import os
from torch.autograd import Variable

def set_random_seed(seed=1234):
    """
    set seed for reproduction
    :param seed:default=1234
    :return:None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def total_time(func):
    """
    Calculate the running time of the specified function
    :param func: the function need to been calculated
    :return: run time
    """
    def call_fun(*args, **kwargs):
        start_time = time.time()
        f = func(*args, **kwargs)
        end_time = time.time()
        print('%s() run time：%s s' % (func.__name__, float(end_time - start_time)))
        return f
    return call_fun

def calculate_model_parm_nums(model):            # 得到模型参数总量
    """
    calculate the number of model parameters
    :param model: defined model
    :return: the num of ('total'/'weight')params
    """
    # temp = [param.nelement() for name, param in model.named_parameters() if ('fc' in name or 'conv' in name) and 'weight' in name]
    # temp = [param.nelement() for name, param in model.named_parameters() if('conv' in name or 'classifier' in name) and 'bias' not in name]
    # total = sum(temp)
    total = sum([param.nelement() for param in model.parameters()])
    if total/1e6 < 1:
        print('Number of params is %dK' % int((total / 1e3)))
    else:
        print('Number of params is %fM' % (total / 1e6))     # 每一百万为一个单位
    return total

def evaluate_trained_model_precision(model, testLoader):
    """
    usage:
    teacher_model = resnet.__dict__['resnet56']().cuda()
    _model = torch.load('/home/huzhou/paperCode/temporary_experiments/ttq-resnet-cifar10/cifar10-resnet-fp/saved_model/resnet56_94.71.pth')
    teacher_model.load_state_dict(_model)
    Helper.evaluate_trained_model_precision(teacher_model, val_loader)

    evaluate topk-precision of trained model
    :param model: initialize model by load_state_dict()
    :param testLoader: test dataset
    :return: precision of trained model
    """
    model.eval()
    correctClass1 = 0
    correctClass5 = 0
    totalNumExamples = 0
    for idx_minibatch, data in enumerate(testLoader):
        inputs, labels = data

        with torch.no_grad():
            inputs = Variable(inputs)
        labels = Variable(labels)

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # model=model.cuda()
        outputs = model(inputs)
        _, top1_predictions = outputs.topk(1, dim=1, largest=True, sorted=True)
        _, top5_predictions = outputs.topk(5, dim=1, largest=True, sorted=True)

        top1_predictions = top1_predictions.t().cpu()
        top5_predictions = top5_predictions.t().cpu()
        correct1 = top1_predictions.eq(labels.view(1, -1).expand_as(top1_predictions))
        correct5 = top5_predictions.eq(labels.view(1, -1).expand_as(top5_predictions))

        correctClass1 += correct1.view(-1).float().sum(0, keepdim=True).data[0]
        correctClass5 += correct5.view(-1).float().sum(0, keepdim=True).data[0]
        totalNumExamples += len(labels)

    top1_accuracy = correctClass1 / totalNumExamples
    top5_accuracy = correctClass5 / totalNumExamples
    print("Test top-1 accuracy of trained model: {test_accuracy:.2f}%".format(test_accuracy=top1_accuracy*100.))
    print("Test top-5 accuracy of trained model: {test_accuracy:.2f}%".format(test_accuracy=top5_accuracy*100.))

def initialize_model_weight_and_bias(model):
    """
    initialize the parameters of the given model
    :param model: given network model include Linear layer or Conv layer and so on
    :return: initialized model
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.xavier_uniform_()
            m.bias.data.fill_(0)

        if isinstance(m, nn.Conv2d):
            # you can define own initialization function here
            pass

def save_best_prec_trained_model_params(model, save_name, best_precision=None, save_path='./saved_model', epoch=0):
    """
    all parameters of the model are saved when this model reaches its best accuracy
    :param model: trained model that its precision is best
    :param save_path: path/to/stored-file
    :param save_name: stored-file name
    :return: ./saved_model/save_name_xx.pth (xx is best valid accuracy)
    """
    def remove_low_precision_model(save_name, save_path, best_precision):
        _file_list = os.listdir(save_path)
        _len = len(save_name)+1
        for f in _file_list:
            if save_name in f:
                try:
                    _precision = float(f[_len:-4])
                    if _precision < best_precision:
                        os.remove(os.path.join(save_path,f))
                except:
                    return

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if best_precision:
        remove_low_precision_model(save_name, save_path, best_precision)
        _file_name = save_name + '_' + str(best_precision)[:6] + '_' + str(epoch) + '.pth'
    else:
        _file_name = save_name

    torch.save(model.state_dict(), os.path.join(save_path, _file_name))

def save_all_prec_trained_model_params(model, save_name, val_prec1, save_path='./saved_model'):
    """
    all parameters of the model are saved when this model reaches its best accuracy
    :param model: trained model that its precision is best
    :param save_path: path/to/stored-file
    :param save_name: stored-file name
    :return: ./saved_model/save_name_xx.pth (xx is best valid accuracy)
    """
    # def remove_low_precision_model(save_name, save_path, best_precision):
    #     _file_list = os.listdir(save_path)
    #     _len = len(save_name)+1
    #     for f in _file_list:
    #         if save_name in f:
    #             try:
    #                 _precision = float(f[_len:-4])
    #                 # if _precision < best_precision:
    #                 #     os.remove(os.path.join(save_path,f))
    #             except:
    #                 return

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # if best_precision:
    #     remove_low_precision_model(save_name, save_path, best_precision)
    #     _file_name = save_name + '_' + str(best_precision)[:6]+'.pth'
    # else:
    _file_name = save_name+'_'+ str(val_prec1)[:6]+'.pth'

    torch.save(model.state_dict(), os.path.join(save_path, _file_name))

def save_checkpoint_state(model, optimizer, epoch, best_prec1, best_prec5, best_epoch, save_path='./saved_model'):
    """
    saving the state of checkpoint for resuming training
    :param model: the trained model in the current epoch
    :param epoch: the current epoch
    :param best_prec1: the best prec1 in the current epoch
    :param best_prec5: the best prec5 in the current epoch
    :param best_epoch: the epoch of getting the best prec1
    :param save_path: path/to/the saved checkpoint
    :return:None
    """
    def remove_old_checkpoint(current_epoch, save_path):
        _file_list = os.listdir(save_path)
        _len = len('checkpoint')+1
        for f in _file_list:
            if 'checkpoint' in f:
                _epoch = int(f[_len:-4])
                if _epoch < current_epoch:
                    os.remove(os.path.join(save_path, f))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    _name = 'checkpoint_' + str(epoch) + '.pth'
    filename = os.path.join(save_path, _name)
    checkpoint_state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'best_prec1': best_prec1,
        'best_prec5': best_prec5,
        'best_epoch': best_epoch,
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint_state, filename)
    remove_old_checkpoint(epoch, save_path)

def get_checkpoint_state(saved_path='./saved_model'):
    ck_state = None
    try:
        _file_list = os.listdir(saved_path)
        for f in _file_list:
            if 'checkpoint' in f:
                ck_state = torch.load(os.path.join(saved_path, f))
                return ck_state
    except Exception:
        return ck_state

def save_best_prec_trained_model_params_for_imagenet(model, save_name, best_prec1, best_prec5, save_path='./saved_model'):
    """
    all parameters of the model(for imagenet experiment) are saved when this model reaches its best accuracy
    :param model: trained model that its precision is best
    :param save_path: path/to/stored-file
    :param save_name: stored-file name
    :param best_prec1: best validation prec1
    :param best_prec5: best validation prec5
    :return: ./saved_model/save_name_xx_yy.pth (xx is best validation prec1, yy is best validation prec5)
    """
    def remove_low_precision_model(save_name, save_path, best_prec1):
        _file_list = os.listdir(save_path)
        _len = len(save_name)+1
        for f in _file_list:
            if save_name in f:
                _prec1 = float(f[_len:_len+6])
                if _prec1 < best_prec1:
                    os.remove(os.path.join(save_path,f))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if best_prec1:
        remove_low_precision_model(save_name, save_path, best_prec1)
        _file_name = save_name + '_' + str(best_prec1)[:6]+ '_' + str(best_prec5)[:6]+'.pth'
    else:
        _file_name = save_name
    torch.save(model.state_dict(), os.path.join(save_path, _file_name))

def get_trained_model_params(file_name):
    """
    get all parameters of the trained model include weight, bias and so on
    :param file_name: path/to/saved_model_name.pth
    :return: the parameters you need
    """
    weight_list = []
    model_params = torch.load(file_name)
    for param in model_params:
        # you can get all params you need here by this way
        if 'weight' in param:
            weight_list.append(model_params[param])
    return weight_list

# def save_train_loss(loss,save_name='loss.json'):
#     """
#     save trian loss to json file
#     :return:
#     """
#     with open(save_name, 'w') as file_obj:
#         json.dump(loss, file_obj)