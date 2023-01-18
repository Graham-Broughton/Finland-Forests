import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import rasterio
import numpy as np


class SentinelDataset2(Dataset):
    def __init__(self, annotations_file, img_dir, train=True, transform=None, target_transform=None):
        self.dataframe = pd.read_csv(annotations_file)
        self.chip_ids = self.dataframe.chip_id.unique()
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.chip_ids = self.dataframe[self.dataframe['split'] == 'train']['chip_id']
        else:
            self.chip_ids = self.dataframe[self.dataframe['split'] == 'test']['chip_id']
        

    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.chip_ids[idx] + "_S2_06.tif")
        image = torch.tensor(rasterio.open(img_path).read().astype(np.float32)[:10])
        # Normalize image with mean 0 and stddev 1. Add a little bit to div to avoid dividing by 0
        image = (image.permute(1,2,0) - image.mean(dim=(1,2)))/(image.std(dim=(1,2)) + 0.01)
        image = image.permute(2,0,1)
       
        label_path = os.path.join(self.img_dir, self.chip_ids[idx] + "_agbm.tif")
        label = torch.tensor(rasterio.open(label_path).read().astype(np.float32))

        if self.transform:
            image = self.transform(image)
            
        return image, label
