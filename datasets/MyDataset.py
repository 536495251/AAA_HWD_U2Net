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
    def __getitem__(self, index):
        img_path = osp.join(self.data.image_path, self.data.images_name_list[index] + self.data.image_category)
        mask_path = osp.join(self.data.mask_path, self.data.images_name_list[index] + self.data.image_category)
        img, mask = self.data.data_transform(Image.open(img_path).convert('RGB'), Image.open(mask_path))
        img, mask = self.data.transform(img), transforms.ToTensor()(mask)
        return img, mask
        # laplacian_tensor, binary_tensor = self.LoG_transform(img)
        # img, mask = self.data.transform(img), transforms.ToTensor()(mask)
        # img = torch.cat([img, laplacian_tensor, binary_tensor], dim=0)  # [5,H,W]
        # return img, mask
    def __len__(self):
        return self.data.data_len

    def LoG_transform(self, image, sigma=2.0, ksize=5, thresh_ratio=0.5):
        # 确保是 PIL.Image 或 numpy 数组
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # 如果是彩色图，转为灰度
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.astype(np.float32)  # 保证浮点运算

        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
        laplacian = -laplacian  # 对亮斑检测取反

        lap_min, lap_max = laplacian.min(), laplacian.max()
        if lap_max - lap_min != 0:
            laplacian_norm = (laplacian - lap_min) / (lap_max - lap_min)
        else:
            laplacian_norm = laplacian - lap_min

        laplacian_norm = laplacian_norm * 2.0 - 1.0  # [-1,1]

        thresh_value = lap_min * thresh_ratio
        _, binary_map = cv2.threshold(-laplacian, -thresh_value, 255, cv2.THRESH_BINARY)
        binary_map = binary_map.astype(np.uint8)

        binary_map_norm = binary_map / 255.0
        binary_map_norm = binary_map_norm * 2.0 - 1.0  # [-1,1]

        laplacian_tensor = torch.from_numpy(laplacian_norm).float().unsqueeze(0)
        binary_tensor = torch.from_numpy(binary_map_norm).float().unsqueeze(0)

        return laplacian_tensor, binary_tensor