import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
from PIL import Image
from datasets.DataInit import DataInit
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, dataset,mode):
        self.data=DataInit(dataset,mode)
        self.add_name=''
        if dataset=='NUAA-SIRST':
            self.add_name='_pixels0'
    def __getitem__(self, index):
        img_path = osp.join(self.data.image_path, self.data.images_name_list[index] + self.data.image_category)
        mask_path = osp.join(self.data.mask_path, self.data.images_name_list[index] + self.add_name+self.data.image_category)
        img, mask = self.data.data_transform(Image.open(img_path).convert('RGB'), Image.open(mask_path))
        img, mask = self.data.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return self.data.data_len
