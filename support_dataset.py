import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as tt
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


class CustomImageDataset(Dataset):
    def __init__(self, info_file, img_dir, use_lab_na, whe_binary, sam_rate, transform=None, target_transform=None):
        self.info = sample(use_lab_na, pd.read_excel(info_file), sam_rate)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.use_lab_na = use_lab_na
        self.whe_binary = whe_binary

        # get column names
        al_col = list(self.info.columns)
        self.fi_pa_na = 'filename'
        self.lab_na = ['重症肺炎', '肺炎+呼吸衰竭', '肺炎+低血氧症', '肺炎+胸腔积液', '肺炎+肺不张']
        self.oth_na = ['visitnum']
        self.fea_na = [x for x in al_col if x not in [self.fi_pa_na] + self.lab_na + self.oth_na]


    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        # read image
        img_path = os.path.join(self.img_dir, self.info.iloc[idx][self.fi_pa_na])
        image = read_image(img_path)
        image = torch.concat([image, image, image], 0).float()

        # get feature
        fea = self.info.iloc[idx][self.fea_na]
        fea = np.array(fea).astype(np.float32)
        fea = torch.tensor(fea, dtype=torch.float)

        # get label
        label = self.info.iloc[idx][self.use_lab_na]
        label = np.array(label).astype(np.int32)

        if self.whe_binary == 'binary':
            label = np.max(label)
            label = torch.tensor(label, dtype=torch.int64)
        elif self.whe_binary == 'nobinary':
            label = torch.tensor(label, dtype=torch.float)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, fea, label

def get_transforms():
    transforms = torch.nn.Sequential(
        tt.Resize((256, 256)),
        tt.RandomCrop(size=224),
        tt.RandomRotation(degrees=(-10, 10), expand=False),
        tt.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        tt.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.12), scale=(0.9, 0.99)),
        tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    return transforms

def transforms_for_HM():
    transforms = torch.nn.Sequential(
        tt.Resize((256, 256)),
        tt.RandomCrop(size=224),
        tt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )
    return transforms

# sample
def sample(use_lab_na, info, sam_rate):
    if sam_rate != -1:
        uind1 = np.array([], dtype=np.int64)
        for na in use_lab_na:
            ind1 = np.array(info.index[info[na] == 1])
            uind1 = np.union1d(ind1, uind1)

        iind2 = np.arange(len(info))
        for na in ['重症肺炎', '肺炎+呼吸衰竭', '肺炎+低血氧症', '肺炎+胸腔积液', '肺炎+肺不张']:
            ind2 = np.array(info.index[info[na] == 0])
            iind2 = np.intersect1d(iind2, ind2)

        sel_iind2 = np.random.choice(iind2, int(len(uind1) * sam_rate), replace=False)
        all_ind = np.concatenate([uind1, sel_iind2])
        sel_info = info.iloc[all_ind]
    else:
        sel_info = info
    return sel_info