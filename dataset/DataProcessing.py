from torch.utils.data import Dataset
from PIL import Image
import glob

IMG_SIZE1 = 480
IMG_SIZE2 = 320


def default_loader(path):
    x = Image.open(path).convert('RGB')
    return x.resize((IMG_SIZE1, IMG_SIZE2), Image.LANCZOS)


def default_loader1(path):
    x = Image.open(path).convert('L')
    return x.resize((IMG_SIZE1, IMG_SIZE2), Image.LANCZOS)


class RoadDataset(Dataset):
    def __init__(self, ImgPath, GTPath, transform=None, ImgLoader=default_loader, GTLoader=default_loader1):
        self.ImgList = sorted(glob.glob(ImgPath + '*.png'))
        self.GTList = sorted(glob.glob(GTPath + '*.png'))
        self.transform = transform
        self.ImgLoader = ImgLoader
        self.GTLoader = GTLoader

    def __getitem__(self, item):
        Img_FN = self.ImgList[item]
        GT_FN = self.GTList[item]
        Img = self.ImgLoader(Img_FN)
        GT = self.GTLoader(GT_FN)

        if self.transform is not None:
            Img = self.transform(Img)
            GT = self.transform(GT)

        return Img, GT

    def __len__(self):
        return len(self.ImgList)
