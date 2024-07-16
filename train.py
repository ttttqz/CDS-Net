import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.DataProcessing import *
from module.CDS import CDSNET
import time
from utils.f1score import calculate_f_measure
from loss.loss import exfloss
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ImgPath = r'G:\\dataset\\BJN260_resize\\img\\'
GTPath = r'G:\\dataset\\BJN260_resize\\gt\\'

TrainingSet = RoadDataset(ImgPath=ImgPath, GTPath=GTPath, transform=transforms.ToTensor())
TrainingLoader = DataLoader(dataset=TrainingSet, batch_size=2, shuffle=True, drop_last=True)

valImgPath = r"G:\\dataset\\BJN260_resize\\val\\"
valGTPath = r"G:\\dataset\\BJN260_resize\\val_gt\\"

ValSet = RoadDataset(ImgPath=valImgPath, GTPath=valGTPath, transform=transforms.ToTensor())
ValLoader = DataLoader(dataset=ValSet, batch_size=1, shuffle=False)


Network = CDSNET(in_channels=3, out_channels=1)


if torch.cuda.is_available():
    Network = Network.cuda()

optimizer = torch.optim.Adam(Network.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
start_time = time.time()
print('======================Begin Training===============================')
Loss_Sum = []
best_score = 0
for epoch in range(5):
    SLoss = 0.0
    Network.train()
    for i, (Img, GT) in enumerate(TrainingLoader):
        if torch.cuda.is_available():
            Img = Img.cuda()
            GT = GT.cuda()

        Mask = Network(Img)
        Mask = Mask.float()
        GT[GT > 0] = 1
        GT = GT.float()
        STerm = exfloss(Mask, GT)
        SLoss = SLoss + STerm
        optimizer.zero_grad()
        STerm.backward()
        optimizer.step()

    scheduler.step()
    print('epoch=' + str(epoch + 1) + ', SLoss=' + str(SLoss.detach().cpu().numpy()))

    Network.eval()
    with torch.no_grad():
        all_pred_edges = []
        all_gt_edges = []
        for data in ValLoader:
            img, gt = data
            if torch.cuda.is_available():
                img = img.cuda()
                gt = gt.cuda()
            gt[gt > 0] = 1
            mask = Network(img)
            all_pred_edges.extend(mask.cpu().numpy())
            all_gt_edges.extend(gt.cpu().numpy())

        # 第一步：计算ODS
        f_max = 0
        best_threshold = 0
        for threshold in np.linspace(0, 1, num=101):  # 假设我们有100个阈值点评估
            f_total = 0
            for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
                precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
                f_total += f_measure
            f_avg = f_total / len(all_pred_edges)
            if f_avg > f_max:
                f_max = f_avg
        ods_score = f_max
        print("验证集ODS_F1:", ods_score)

        # 第二步：计算OIS
        ois_scores = []
        for pred_edges, gt_edges in zip(all_pred_edges, all_gt_edges):
            f_max = 0
            for threshold in np.linspace(0, 1, num=101):
                precision, recall, f_measure, _ = calculate_f_measure(pred_edges, gt_edges, threshold)
                if f_measure > f_max:
                    f_max = f_measure
            ois_scores.append(f_max)
        ois_score = np.mean(ois_scores)
        print("验证集OIS_F1:", ois_score)
        if ods_score + ois_score > best_score:
            best_score = ods_score + ois_score
            filename = f"./best.pth"
            torch.save(Network.state_dict(), './' + filename)
            print('Model has been successfully saved in this epoch!')
    print('=============================================')

end_time = time.time()
running_time = end_time - start_time
print("代码运行时间为：", running_time, "秒")

