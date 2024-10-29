# quantize activation from dorefanet:https://arxiv.org/pdf/1606.06160.pdf
# code is implemented by https://github.com/666DZY666/micronet/tree/master/micronet/compression/quantization/wqaq/dorefa
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class ActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(ActivationQuantizer, self).__init__()
        self.a_bits = a_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        if self.a_bits == 32:
            output = input
        # elif self.a_bits == 1:
        #     print("！Binary quantization is not supported ！")
        #     assert self.a_bits != 1
        else:
            # output = torch.clamp(input * 0.1, 0, 1)
            output = torch.clamp(input, 0, 1)       #将激活值截断为0到1之间
            scale = 1 / float(2 ** self.a_bits - 1)  # scale
            output = self.round(output / scale) * scale  # 量化/反量化
        return output