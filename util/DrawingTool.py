"""
This class is used for drawing some related statistical chart
usage:
1.from util.DrawingTool import DrawingTool
2.python trainer_test.py
3.cd save_path('./result') then tensorboard --logdir .
"""
import os
import torch.nn as nn
from tensorboardX import SummaryWriter

class DrawingTool(object):
    def __init__(self, save_path='./result', draw_flag=True):
        self.save_path = save_path
        self.draw_flag = draw_flag
        self.clear_invalid_files()

        self.writer = SummaryWriter(self.save_path, filename_suffix='log')

    def clear_invalid_files(self):
        if os.path.exists(self.save_path):
            _filelist = os.listdir(self.save_path)
            for f in _filelist:
                os.remove(os.path.join(self.save_path,f))


    def draw_weight_histogram(self, model, current_epoch, draw_frequency=20):
        """
        draw nn.Linear and nn.Conv2d layers' weight histograms of the given model every 'draw_frequency' epochs
        :param model: the given model which
        :param current_epoch: current train epoch
        :param draw_frequency: frequency of drawing histograms
        :return: None
        """
        if self.draw_flag and current_epoch==1 or current_epoch%draw_frequency == 0:
            num_linear, num_conv2d = 0, 0
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    num_linear +=1
                    _name = 'Linear-'+str(num_linear)
                    self.writer.add_histogram(_name, m.weight.clone().cpu().data.numpy(), global_step=current_epoch)
                if isinstance(m, nn.Conv2d):
                    num_conv2d +=1
                    _name = 'Conv2d-'+str(num_conv2d)
                    self.writer.add_histogram(_name, m.weight.clone().cpu().data.numpy(), global_step=current_epoch)

    def draw_loss_curve(self, epoch, loss_current_epoch, tag='Train/Loss'):
        if self.draw_flag:
            self.writer.add_scalar(tag, loss_current_epoch, epoch)

    def draw_accuracy_curve(self, epoch, acc_current_epoch, tag='Test/Acc'):
        if self.draw_flag:
            self.writer.add_scalar(tag, acc_current_epoch, epoch)