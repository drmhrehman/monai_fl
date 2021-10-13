import torch
from monai.metrics import compute_meandice

def mean_dice(output, target, average=True):
    # empty labels return a dice score of NaN - replace with 0
    dice_per_batch = compute_meandice(output,target, include_background=False)
    dice_per_batch[dice_per_batch.isnan()] = 0
    if average:
        return dice_per_batch.mean().cpu()
    else:
        return dice_per_batch.cpu()

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
