import os 
import cv2
import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split 
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor
from torchvision import transforms 
import random 
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm


def get_transforms(phase, width=1600, height=256):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                albu.HorizontalFlip(),
                albu.OneOf([
                    albu.RandomContrast(),
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                    ], p=0.3),
                albu.OneOf([
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albu.GridDistortion(),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3),
                albu.ShiftScaleRotate(),
            ]
        )
    list_transforms.extend(
        [
            albu.Resize(width,height,always_apply=True),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensor(),
        ]
    )
    list_trfms = albu.Compose(list_transforms)
    return list_trfms
class SteelDataset(Dataset):
    def __init__(self, root_dir, df, phase):
        super(SteelDataset, self).__init__()
        self.root_dir = root_dir
        self.df = df
        self.transforms = get_transforms(phase, width = 672, height = 1120)
    
    def __read_file__(self, list_data):
        df = pd.read_csv(os.path.join(list_data))
        df['ImageId'], df['ClassId'] = zip(*df['Image_Label'].str.split('_'))
        df['ClassId'] = df['ClassId']
        df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
        df['defects'] = df.count(axis=1)
        return df
    def make_mask(self, row_id, df):
        fname = df.iloc[row_id].name
        labels = df.iloc[row_id][:4]
        masks = np.zeros((1400, 2100, 4), dtype=np.float32)

        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                mask = np.zeros(1400 * 2100, dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos:(pos + le)] = 255
                masks[:, :, idx] = mask.reshape(1400, 2100, order='F')
        return fname, masks
    def __getitem__(self, index):
        image_id, mask = self.make_mask(index, self.df)
        image_path = os.path.join(self.root_dir, image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask'] # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1) # 1x4x256x1600
        return img, mask

    def __len__(self):
        # return len(self.df)
        return 30