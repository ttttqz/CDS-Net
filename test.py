import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.DataProcessing import *
from module.CDS import CDSNET
from utils.f1score import calculate_f_measure
from torchvision import transforms

ImgPath = r'G:\\dataset\\BJN260_resize\\test\\'
GTPath = r'G:\\dataset\\BJN260_resize\\test_gt\\'

TestingSet = RoadDataset(ImgPath=ImgPath, GTPath=GTPath, transform=transforms.ToTensor())
TestingLoader = DataLoader(dataset=TestingSet, batch_size=1, shuffle=False, drop_last=True)

Network = CDSNET(in_channels=3, out_channels=1)
Network.load_state_dict(torch.load('./best.pth'))

if torch.cuda.is_available():
    Network = Network.cuda()

Network.eval()

visual = 0  # 可视化

if visual == 1:
    for step, (Img, GT) in enumerate(TestingLoader):
        print('==============' + str(step) + '===============')
        if torch.cuda.is_available():
            Img = Img.cuda()
            GT = GT.cuda()
        GT[GT > 0] = 1
        Mask = Network(Img)
        Mask = Mask.float()
        Img = Img.float()
        Mask = torch.squeeze(Mask)
        Img = torch.squeeze(Img)
        GT = torch.squeeze(GT)
        Img = Img.permute(1, 2, 0)
        Img = Img.detach().cpu().numpy()
        Mask = Mask.detach().cpu().numpy()
        GT = GT.detach().cpu().numpy()
        best_threshold = 0.3  # ODS best_threshold
        Mask[Mask >= best_threshold] = 1
        Mask[Mask < best_threshold] = 0

        cv2.imshow("Img", Img)
        cv2.imshow("GT", GT)
        cv2.imshow("Mask", Mask)

        cv2.waitKey(0)

# 存储所有的预测边缘和真实边缘
all_pred_edges = []
all_gt_edges = []

with torch.no_grad():
    for data in TestingLoader:
        # 假设边缘被存储在batch中的'targets'键中, 输入图片在'inputs'键中
        img, gt = data
        # 将数据移动到模型所在设备上
        if torch.cuda.is_available():
            img = img.cuda()
            gt = gt.cuda()
        gt[gt > 0] = 1
        # 运行模型得到预测结果
        mask = Network(img)
        # 将预测和真实边缘图转移到CPU并添加到列表中
        all_pred_edges.extend(mask.cpu().numpy())
        all_gt_edges.extend(gt.cpu().numpy())

# 第一步：计算ODS
f_max = 0
pscore = 0
rscore = 0
best_threshold = 0
for threshold in np.linspace(0, 1, num=101):  # 假设我们有100个阈值点评估
    f_total = 0
    p_total = 0
    r_total = 0
    for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
        precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
        p_total += precision
        r_total += recall
        f_total += f_measure
    p_avg = p_total / len(all_pred_edges)
    r_avg = r_total / len(all_pred_edges)
    f_avg = f_total / len(all_pred_edges)
    if f_avg > f_max:
        f_max = f_avg
        pscore = p_avg
        rscore = r_avg
        best_threshold = threshold
ods_score = f_max
print("ODS_P:", pscore)
print("ODS_R:", rscore)
print("ODS_F1:", ods_score)

# 第二步：计算OIS
ois_scores = []
p_scores = []
r_scores = []
for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
    f_max = 0
    pscore = 0
    rscore = 0
    for threshold in np.linspace(0, 1, num=101):
        precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
        if f_measure > f_max:
            f_max = f_measure
            pscore = precision
            rscore = recall
    ois_scores.append(f_max)
    p_scores.append(pscore)
    r_scores.append(rscore)

ois_score = np.mean(ois_scores)
p_scores = np.mean(p_scores)
r_scores = np.mean(r_scores)
print("OIS_P:", p_scores)
print("OIS_R:", r_scores)
print("OIS_F1:", ois_score)

print("best_threshold:", best_threshold)

miou_scores = []
for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
    _, _, _, iou = calculate_f_measure(pred_edges, gt_edges, best_threshold)
    miou_scores.append(iou)
miou = np.mean(miou_scores)

print("mIoU:", miou)
