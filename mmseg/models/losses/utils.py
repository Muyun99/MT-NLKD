# Copyright (c) OpenMMLab. All rights reserved.
import functools

import mmcv
import torch
import numpy as np
import torch.nn.functional as F


def get_class_weight(class_weight):
    """Get class weight for loss function.

    Args:
        class_weight (list[float] | str | None): If class_weight is a str,
            take it as a file name and read from it.
    """
    if isinstance(class_weight, str):
        # take it as a file path
        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:
            # pkl, json or yaml
            class_weight = mmcv.load(class_weight)

    return class_weight

def map_to_one_hot(x, num_class):
    assert x.dim() == 3
    x = x.where((x >= 0) & (x < num_class), torch.tensor([num_class], dtype=x.dtype, device=x.device))

    B, H, W = x.shape
    x_onehot = x.new_zeros((B, num_class + 1, H, W), dtype=torch.float).cuda()
    x_onehot = x_onehot.scatter_(dim=1, index=x.long().view(B, 1, H, W), value=1)[:, :-1, :, :]

    return x_onehot.contiguous()


def compute_edge(seg):
    # seg: (B, C, H, W)

    seg = seg.float()

    avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)

    seg_avg = avgpool(seg.float())

    seg = (seg != seg_avg).float() * seg

    return seg


def dilate(seg, tol):
    # seg: (B, C, H, W)

    maxpool = torch.nn.MaxPool2d(2 * int(tol) + 1, stride=1, padding=int(tol))

    seg = maxpool(seg.float())

    return seg

def bfscore_val(output, target, num_class, tol):
    # output: (B, H, W)
    # target_not_onehot: (B, H, W)

    N, H, W = target.shape

    target = map_to_one_hot(target, num_class)
    output = map_to_one_hot(output, num_class)

    output_edge = compute_edge(output)
    target_edge = compute_edge(target)

    if tol > 0:
        target_dilate = dilate(target_edge, tol)
        output_dilate = dilate(output_edge, tol)
    else:
        target_dilate = target_edge
        output_dilate = output_edge

    # output_dilate: (B, C, H, W)
    # target_dialte: (B, C, H, W)
    output_edge = output_edge
    target_edge = target_edge

    n_gt = target_edge.sum(dim=3).sum(dim=2).sum(dim=0)
    n_fg = output_edge.sum(dim=3).sum(dim=2).sum(dim=0)

    match_fg = torch.sum(output_edge * target_dilate, dim=3).sum(dim=2).sum(dim=0)
    match_gt = torch.sum(output_dilate * target_edge, dim=3).sum(dim=2).sum(dim=0)

    p = match_fg / (n_fg + 1e-8)
    r = match_gt / n_gt

    f1 = 2 * p * r / (p + r + 1e-8)

    return f1, n_fg, n_gt, match_fg, match_gt

def mIoU_eval(target, output, num_class):
    confusionMatrix = np.zeros((num_class,) * 2)
    assert output.shape == target.shape
    mask = (target)
    # if trained_id is not None:
    #     for id in trained_id:
    #         mask = (target == id) & mask
    # else:
    mask = (target >= 0) & (target < num_class)
    label = num_class * target[mask] + output[mask]
    count = np.bincount(label, minlength=num_class ** 2)
    confusionMatrix = count.reshape(num_class, num_class)

    intersection = np.diag(confusionMatrix)  # 取对角元素的值，返回列表
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(
        confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
    IoU = intersection / union  # 返回列表，其值为各个类别的IoU
    mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
    return mIoU

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        weight = weight.unsqueeze(1).unsqueeze(1)
        mean_weight = weight.mean()
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight
        loss = loss / mean_weight
    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
