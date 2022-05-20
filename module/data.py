import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import module.utils as utils

def Horizontal_Vertical_filp():


class CfImageDataset(Dataset):
    """create a Image Dataset for common classification"""
    def __init__(self, annotations_file, img_dir, mode: str, valid_rate=0.1, transform=None, target_transform=None):
        self.img_csv = pd.read_csv(annotations_file, header=None)
        self.img_label_dict = utils.label_convert(annotations_file, 'num')

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        self.data_len = len(self.img_csv.index) - 1
        self.data_len = int(self.data_len * (1 - valid_rate))

        if mode == 'train':
            self.img_csv = self.img_csv.iloc[1:self.data_len, :]
            self.data_len = len(self.img_csv.index)
        elif mode == 'valid':
            self.img_csv = self.img_csv.iloc[self.data_len:, :]
            self.data_len = len(self.img_csv.index)
        elif mode == 'test':
            self.data_len = len(self.img_csv.index) - 1

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        if self.mode == "train" or "valid":
            img_path = os.path.join(self.img_dir,
                                    self.img_csv.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_label_dict[self.img_csv.iloc[idx, 1]]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image.float(), label

        elif self.mode == "test":
            img_path = os.path.join(self.img_dir,
                                    self.img_csv.iloc[idx, 0])
            image = read_image(img_path)
            return image.float()