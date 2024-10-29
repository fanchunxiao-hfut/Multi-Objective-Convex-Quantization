import torch
import torch.nn as nn
import math

### Our designed convex quantization loss

epsilon = 1e-2

class LBClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_value, max_value):
        return torch.clamp(input, min_value, max_value)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def constraint_qc(x):
    y = -(torch.log(torch.abs(torch.cos(x))))
    return y



class QC_Loss(nn.Module):
    def __init__(self, scale, QM, QN):
        super(QC_Loss, self).__init__()
        self.clamp = LBClamp.apply
        self.s = scale
        self.QM = QM
        self.QN = QN
        
    def forward(self, x):
        min_value = self.QM - self.s / 2 + epsilon
        max_value = self.QN + self.s / 2 - epsilon
        temp = self.clamp(x, min_value, max_value)                          
        qc = torch.sum(constraint_qc(temp/self.s * math.pi))
        return qc