"""
usage:
from util.CalculateTool import AverageMeter,calculate_topK_accuracy
"""
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_topK_accuracy(output, target, topk=(1,)):
    """
    calculate the precision@k for the specified values of k
    :param output: the output of the model
    :param target: the target of the network input
    :param topk: specified top-k accuracy
    :return: top(1,5) accuracy, like [top1,top5]
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res