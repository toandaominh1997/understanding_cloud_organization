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
from modules.encoders import get_preprocessing_fn


def get_transforms(phase, width=1600, height=256):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                albu.GridDistortion(p=0.5),
                albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),    
            ]
        )
    list_transforms.extend(
        [
            albu.Resize(width, height),
            # albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            # ToTensor(),
        ]
    )
    list_trfms = albu.Compose(list_transforms)
    return list_trfms
class SteelDataset(Dataset):
    def __init__(self, root_dir, df, img_ids,  phase, encoder_name, pretrained):
        super(SteelDataset, self).__init__()
        self.root_dir = root_dir
        self.df = df
        self.img_ids = img_ids
        self.transforms = get_transforms(phase, width = 320, height = 640)
        self.preprocessing = self.get_preprocessing(get_preprocessing_fn(encoder_name, pretrained))

    
    @staticmethod
    def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
        '''
        Decode rle encoded mask.

        :param mask_rle: run-length as string formatted (start length)
        :param shape: (height, width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape, order='F')
    def make_mask(self, df: pd.DataFrame, image_name: str='img.jpg', shape: tuple = (1400, 2100)):
        """
        Create mask based on df, image name and shape.
        """
        encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
        masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

        for idx, label in enumerate(encoded_masks.values):
            if label is not np.nan:
                mask = self.rle_decode(label)
                masks[:, :, idx] = mask
                
        return masks
    
    @staticmethod
    def to_tensor(x, **kwargs):
        """
        Convert image or mask.
        """
        return x.transpose(2, 0, 1).astype('float32')
    def get_preprocessing(self, preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
            ]
        return albu.Compose(_transform)
    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = self.make_mask(self.df, image_name)
        image_path = os.path.join(self.root_dir, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        preprocessed = self.preprocessing(image=img, mask=mask)
        img = preprocessed['image']
        mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.df)