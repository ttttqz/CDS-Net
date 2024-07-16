import numpy as np


def calculate_f_measure(pred_edges, gt_edges, threshold):
    # 将预测概率转换为二值边缘图
    binary_pred_edges = (pred_edges >= threshold).astype(int)

    # 计算混淆矩阵：TP, FP, FN
    tp = np.logical_and(binary_pred_edges == 1, gt_edges == 1).sum()  # True Positive
    fp = np.logical_and(binary_pred_edges == 1, gt_edges == 0).sum()  # False Positive
    fn = np.logical_and(binary_pred_edges == 0, gt_edges == 1).sum()  # False Negative

    # 计算精度和召回率
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

    # F-measure

    f_measure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f_measure, iou
