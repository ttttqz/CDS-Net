import torch


def exfloss(my_preds, gt_preds):
    assert my_preds.shape == gt_preds.shape
    eps = 0.00001
    my_preds = my_preds.clamp(min=eps, max=1.0 - eps)
    alpha = torch.tensor(gt_preds == 0).sum() / torch.tensor(gt_preds == 1).sum()
    alpha = torch.sqrt(alpha)
    beta = 1 + torch.sqrt(alpha)
    y1 = torch.ones_like(gt_preds)
    y0 = torch.zeros_like(gt_preds)

    threshold = 0.9

    w1 = torch.where(my_preds <= threshold, y1, y0)
    w0 = torch.where(my_preds > threshold, y1, y0)

    loss = -alpha * gt_preds * torch.log(my_preds)
    loss -= (1 - gt_preds) * w1 * torch.log(1 - my_preds)
    loss -= (1 - gt_preds) * w0 * beta * torch.log(1 - my_preds)

    loss = loss.mean()
    return loss


def soft_dice_loss(y_pred, y_true, epsilon=1e-6):
    # skip the batch and class axis for calculating Dice score
    numerator = 2. * torch.sum(y_pred * y_true)
    denominator = torch.sum(y_pred + y_true)

    return 1 - torch.mean(numerator / (denominator + epsilon))


def iou_loss(output, target, smooth=1e-6):
    intersection = (output * target).sum()
    total = (output + target).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


def combined_loss(output, target, wce_weight=0.7, iou_weight=0.3):
    wce = exponential_function_loss(output, target)
    iou = iou_loss(output, target)
    total_loss = wce_weight * wce + iou_weight * iou
    return total_loss


def exponential_function_loss(my_preds, gt_preds):
    assert my_preds.shape == gt_preds.shape
    eps = 0.00001
    my_preds = my_preds.clamp(min=eps, max=1.0 - eps)
    alpha = torch.tensor(gt_preds == 0).sum() / torch.tensor(gt_preds == 1).sum()
    alpha = torch.sqrt(alpha)
    loss = -alpha * gt_preds * torch.log(my_preds)
    loss -= (1 - gt_preds) * torch.log(1 - my_preds)
    loss = loss.mean()
    return loss