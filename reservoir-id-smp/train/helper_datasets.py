
import os
from skimage import io
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset as BaseDataset

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform, is_check_shapes=False)

def normalize_image(ar, mean_std):
    return (ar - mean_std[0])/mean_std[1]

class DatasetImageOnly(BaseDataset):
    CLASSES = ['background', 'water']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            mean_std = None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mean_std = mean_std
    
    def __getitem__(self, i):
        
        # read data
        image = io.imread(self.images_fps[i])
        if self.mean_std is not None:
            image = normalize_image(image, self.mean_std)
        
        # IMPORTANT: Remove extra bands
        if image.shape[2] >= 10:
            image = image[:,:,:6]
    
        mask = io.imread(self.masks_fps[i])
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']  

        #Convert to PIL
        return image
        
    def __len__(self):
        return len(self.ids)

class Dataset(BaseDataset):
    
    CLASSES = ['background', 'water']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            mean_std = None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mean_std = mean_std
    
    def __getitem__(self, i):
        
        # read data
        image = io.imread(self.images_fps[i])
        if self.mean_std is not None:
            image = normalize_image(image, self.mean_std)
        
        # IMPORTANT: Remove extra bands
        if image.shape[2] >= 10:
            image = image[:,:,:6]

        mask = io.imread(self.masks_fps[i])
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']  

        #Convert to PIL
        return {'image':image, 'mask':mask}
        
    def __len__(self):
        return len(self.ids)