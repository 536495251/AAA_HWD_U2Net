
import torch.nn as nn
class SoftIoULoss(nn.Module):

    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss


# def criterion(inputs, target):
#     if isinstance(inputs, list):
#         losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
#         total_loss = sum(losses)
#     else:
#         total_loss = F.binary_cross_entropy_with_logits(inputs, target)
#
#     return total_loss
#——————————————————————————————————————————————————————————————改进——————————————————————————————————————————————————————
import torch
import torch.nn.functional as F

def soft_iou_loss(inputs, targets, smooth=1e-6):
    """Soft IoU loss"""
    inputs = torch.sigmoid(inputs)
    intersection = (inputs * targets).sum(dim=(1,2,3))
    union = (inputs + targets - inputs*targets).sum(dim=(1,2,3))
    loss = 1 - (intersection + smooth) / (union + smooth)
    return loss.mean()

def centroid_loss(inputs, targets):
    """质心距离归一化"""
    inputs = torch.sigmoid(inputs)
    B, _, H, W = inputs.shape
    losses = []

    for b in range(B):
        pred_mask = (inputs[b,0] > 0.5).float()
        gt_mask   = (targets[b,0] > 0.5).float()
        pred_y, pred_x = torch.nonzero(pred_mask, as_tuple=True)
        gt_y, gt_x     = torch.nonzero(gt_mask, as_tuple=True)

        if len(pred_x) > 0 and len(gt_x) > 0:
            cx_pred = pred_x.float().mean() / W
            cy_pred = pred_y.float().mean() / H
            cx_gt   = gt_x.float().mean() / W
            cy_gt   = gt_y.float().mean() / H
            losses.append(((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2)**0.5)
        else:
            losses.append(torch.tensor(0.0, device=inputs.device))

    return torch.stack(losses).mean()

def criterion(inputs, targets, w_iou=1.0, w_centroid=0.5):
    """Soft-IoU + Centroid复合损失"""
    if isinstance(inputs, list):
        total_loss = 0
        for out in inputs:
            iou_loss = soft_iou_loss(out, targets)
            c_loss   = centroid_loss(out, targets)
            total_loss += w_iou*iou_loss + w_centroid*c_loss
        total_loss /= len(inputs)
    else:
        iou_loss = soft_iou_loss(inputs, targets)
        c_loss   = centroid_loss(inputs, targets)
        total_loss = w_iou*iou_loss + w_centroid*c_loss

    return total_loss