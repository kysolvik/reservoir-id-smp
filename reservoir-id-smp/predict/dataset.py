from torch.utils.data import Dataset as BaseDataset
import os
from skimage import io
import numpy as np

def normalize_image(ar, mean_std):
    return (ar - mean_std[0])/mean_std[1]

class ResDataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        imgs (ar):
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)

    """


    def __init__(
            self,
            imgs,
            preprocessing=None,
            mean_std=None
    ):
        self.ids = np.arange(imgs.shape[0])
        self.imgs = imgs
        self.preprocessing = preprocessing
        self.mean_std = mean_std


    def __getitem__(self, i):

        # read data
        image = self.imgs[i]

#        if self.mean_std is not None:
#            image = normalize_image(image, self.mean_std)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return {'image': image}

    def __len__(self):
        return len(self.ids)
