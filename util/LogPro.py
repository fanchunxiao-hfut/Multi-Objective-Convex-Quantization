# -*- coding:utf-8 -*-
# Introduction: LogPro is used to record training log information
# Usage:
# 1.
# import logging
# 2.
# imort util.LogPro as LogPro
# 3.
# We should initialize LogPro in main(),parameter record_flag detemines whether to save the printed log information by logging.info ,
# like that:
# LogPro(record_flag=True)
# we can use logging.info() or logging.debug() to record log everywhere, like that:
# logging.info("...") or loggin.debug("....")

import os
import time
import sys
import logging

class LogPro(object):
    def __init__(self, record_flag=True):
        """
        initialize parametes
        :param foldername: save folder for logs
        :param record_flag: if record_flag is set False , we don't save logs
        """
        self.record_flag = record_flag
        self.date = time.strftime("%Y-%m-%d_") + time.strftime('%H-%M-%S')
        self.foldername = './log/'
        self.filename = os.path.join(self.foldername, self.date + '.txt')
        self.create_saved_floder()
        self.setup_logging()

    def create_saved_floder(self):
        """
        :return:
        """
        if self.record_flag and not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def setup_logging(self):
        """
        setup logging configuration
        Print Level(from low to high) : DEBUG < INFO < WARNING < ERROR < CRITICAL
        """
        if self.record_flag:
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s-%(levelname)s-%(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S",
                                filename=self.filename,
                                filemode='w')
            console = logging.StreamHandler()               # handler用于输出到控制台
            formatter = logging.Formatter('%(message)s')    # handler输出格式
            console.setLevel(logging.DEBUG)
            console.setFormatter(formatter)
            logging.getLogger().addHandler(console)
        else:
            logging.basicConfig(level=logging.DEBUG,
                                format="%(message)s",
                                stream=sys.stderr)

def print_log_every_epoch(num_epoch, lr, duration, epoch, best_epoch, train_loss, val_loss,
                          best_prec1, val_prec):
    """
    print and record the log informatin of each training epoch
    :param num_epoch: the number of epoch
    :param lr: current learn rate
    :param duration: the duration of a epoch
    :param epoch: current epoch(count from 1 by default)
    :param best_epoch: the epoch that has best validation precision
    :param train_loss: the training loss
    :param val_loss: the validation loss
    :param best_prec1: the best validation precision so far
    :param val_prec: the validatin precision at current epoch
    :return: None
    """
    logging.info('\nEpoch {0} of {1} took {2}s\n'
         '  Learning Rate:                  {lr:.6f}\n'
         '  Training Loss:                  {train_loss:.6f} \n'
         '  Best Validation Accuracy        {best_prec1:.6f}%\n'
         '  Best Epoch:                     {3}\n'
         '  Validation Loss:                {val_loss:.6f} \n'
         '  Validation Accuracy:            {val_prec1:.6f}% \n'.format(epoch, num_epoch, duration, best_epoch,
        lr=lr, train_loss=train_loss, val_loss=val_loss, best_prec1=best_prec1*100, val_prec1=val_prec*100))

def print_log_every_epoch_for_imagenet(num_epoch, lr, duration, epoch, best_epoch, train_loss, val_loss,
                          prec1, prec5, best_prec1, best_prec5, val_prec1, val_prec5):
    """
    print and record the log informatin of each training epoch for imagenet experiments
    :param num_epoch: the number of epoch
    :param lr: current learn rate
    :param duration: the duration of a epoch
    :param epoch: current epoch(count from 1 by default)
    :param best_epoch: the epoch that has best validation prec1 and prec5
    :param train_loss: the training loss
    :param val_loss: the validation loss
    :param prec1: the training prec1 at current epoch
    :param prec5: the training prec5 at current epoch
    :param best_prec1: the best validation prec1 so far
    :param best_prec5: the best validation prec5 so far
    :param val_prec1: the validatin prec1 at current epoch
    :param val_prec5: the validatin prec5 at current epoch
    :return:
    """
    h = duration/3600
    s = int(duration - h*3600)
    duration = h + s/3600

    logging.info('\nEpoch {0} of {1} took {2}h\n'
         '  Learning Rate:                           {lr:.6f}\n'
         '  Training Loss:                           {train_loss:.6f} \n'
         '  Training Prec@1,Prec@5:                  {prec1:.6f}%, {prec5:.6f}%\n'
         '  Best Validation Prec@1,Prec@5:           {best_prec1:.6f}%, {best_prec5:.6f}%\n'
         '  Best Epoch:                              {3}\n'
         '  Validation Loss:                         {val_loss:.6f} \n'
         '  Validation Prec@1,Prec@5:                {val_prec1:.6f}%, {val_prec5:.6f}% \n'.format(
        epoch, num_epoch, duration, best_epoch,
        lr=lr, train_loss=train_loss, prec1=prec1*100, prec5=prec5*100, val_loss=val_loss,
        best_prec1=best_prec1*100, best_prec5=best_prec5*100, val_prec1=val_prec1*100, val_prec5=val_prec5*100))

def print_log_every_epoch_for_cifar(num_epoch,  epoch, best_epoch,  val_loss,
                                        best_prec1, best_prec5, val_prec1, val_prec5):
    """
    print and record the log informatin of each training epoch for imagenet experiments
    :param num_epoch: the number of epoch
    :param lr: current learn rate
    :param duration: the duration of a epoch
    :param epoch: current epoch(count from 1 by default)
    :param best_epoch: the epoch that has best validation prec1 and prec5
    :param train_loss: the training loss
    :param val_loss: the validation loss
    :param prec1: the training prec1 at current epoch
    :param prec5: the training prec5 at current epoch
    :param best_prec1: the best validation prec1 so far
    :param best_prec5: the best validation prec5 so far
    :param val_prec1: the validatin prec1 at current epoch
    :param val_prec5: the validatin prec5 at current epoch
    :return:
    """
    # h = duration / 3600
    # s = int(duration - h * 3600)
    # duration = h + s / 3600

    print('\nEpoch {0} of {1}\n'

                 '  Best Validation Prec@1,Prec@5:           {best_prec1:.6f}%, {best_prec5:.6f}%\n'
                 '  Best Epoch:                              {2}\n'
                 '  Validation Loss:                         {val_loss:.6f} \n'
                 '  Validation Prec@1,Prec@5:                {val_prec1:.6f}%, {val_prec5:.6f}% \n'.format(
        epoch, num_epoch, best_epoch,
        val_loss=val_loss,
        best_prec1=best_prec1 * 100, best_prec5=best_prec5 * 100, val_prec1=val_prec1 * 100,
        val_prec5=val_prec5 * 100))