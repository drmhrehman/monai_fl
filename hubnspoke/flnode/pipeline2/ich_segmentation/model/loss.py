import torch.nn.functional as F
from monai.losses import DiceLoss
from torch.nn import BCELoss

def nll_loss(output, target):
    return F.nll_loss(output, target)

class dice_loss(object):
    def __init__(self):
        self.dice = DiceLoss(include_background=False)

    def __call__(self, output, target):
        return self.dice(output, target)

class bce_loss(object):
    def __init__(self):
        self.bce = BCELoss()

    def __call__(self, output, target):
        return self.bce(output, target)