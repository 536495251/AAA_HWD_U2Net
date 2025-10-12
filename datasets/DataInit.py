import numpy as np

from utils.get_config import get_config
import random
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

class DataInit:
    def __init__(self,dataset,mode):
        self.dataset = get_config('dataset')[dataset]
        self.image_path = self.dataset['image_path']
        self.mask_path = self.dataset['mask_path']
        self.mode =mode
        if self.mode == 'train':
            self.index_file=self.dataset['train']
        elif self.mode == 'val':
            self.index_file=self.dataset['val']
        elif self.mode == 'test':
            self.index_file=self.dataset['test']
        else:
            raise ValueError('mode must be train or val or test')
        self.image_category = self.dataset['image_category']
        self.images_name_list = []
        with open(self.index_file, 'r') as f:
            self.images_name_list += [line.strip() for line in f.readlines()]
        self.data_len=len(self.images_name_list)
        self.image_size=self.dataset['image_size']
        self.crop_size=480
        if dataset=='IRSTD-1k':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.28450727, 0.28450724, 0.28450724], [0.22880708, 0.22880709, 0.22880709]),
            ])
        if dataset=='NUAA-SIRST':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.39594397, 0.39577425, 0.39652345] ,[0.28985128, 0.28985047, 0.29031008]),
            ])

    def data_transform(self,img,mask):
        if self.mode == 'train':
            return self.train_transform(img,mask)
        elif self.mode == 'val':
            return self.val_transform(img,mask)
        elif self.mode == 'test':
            return self.test_transform(img,mask)
        else:
            raise ValueError('mode must be train or val or test')


    def train_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        img_size = self.image_size
        # random scale (short edge)
        long_size = random.randint(int(self.image_size * 0.5), int(self.image_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < img_size:
            padh = img_size - oh if oh < img_size else 0
            padw = img_size - ow if ow < img_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = img.size
        x1 = random.randint(0, w - img_size)
        y1 = random.randint(0, h - img_size)
        img = img.crop((x1, y1, x1 + img_size, y1 + img_size))
        mask = mask.crop((x1, y1, x1 + img_size, y1 + img_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask
    def val_transform(self, img, mask):

        outsize = self.image_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))

        return img, mask

    def test_transform(self, img, mask):
        img_size = self.image_size
        img = img.resize((img_size, img_size), Image.BILINEAR)
        mask = mask.resize((img_size, img_size), Image.NEAREST)

        return img, mask

if __name__ == '__main__':
    print(DataInit('IRSTD-1k','train'))
