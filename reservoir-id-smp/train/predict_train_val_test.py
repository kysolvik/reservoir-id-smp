#! /usr/env/bin python

import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import albumentations as albu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torchvision
import subprocess as sp
from scipy import ndimage

# Set params
mean_std = np.load('./data/mean_stds/mean_std_sentinel_v12.npy')
DATA_DIR = './data/reservoirs_10band_v12/'
save_truth=True
TRUE_NPY = './data/preds/reservoirs_10band_masks_train.npy'

# Replace val with train or test to predict on those
x_valid_dir = os.path.join(DATA_DIR, 'img_dir/train')
y_valid_dir = os.path.join(DATA_DIR, 'ann_dir/train')
PRED_NPY = './data/preds/reservoirs_10band_manet_datav12_modelv6_train.npy'
checkpoint_path = './models/best/sentinel_datav12_modelv6.ckpt'

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        tranform: albumentations.Compose
    """
    
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform, is_check_shapes=False)

def normalize_image(ar, mean_std):
    return (ar - mean_std[0])/mean_std[1]

class Dataset(BaseDataset):
    """
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
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
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.replace('.tif', '.png')) for image_id in self.ids]
        
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

class ResModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.MAnet(encoder_name="resnet34", in_channels=in_channels, classes=out_classes,
                               encoder_weights=None,
                                      aux_params=dict(
                                          classes=1,
                                          dropout=0.2
                                      )
        )


        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.crop_transform = torchvision.transforms.CenterCrop(500)

    def forward(self, image):
        mask = self.model(image)[0]
        return self.crop_transform(mask)

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch['image']).sigmoid()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0001)
        return [optim], [torch.optim.lr_scheduler.ExponentialLR(optim, 0.95)]

model =  ResModel.load_from_checkpoint(checkpoint_path, in_channels=10, out_classes=1, arch='MAnet',
                                       encoder_name='resnet34', map_location=torch.device('cpu'))

CLASSES = ['Water']

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    preprocessing=get_preprocessing(),
    classes=CLASSES,
    mean_std=mean_std,
)

valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2)

trainer = pl.Trainer()
preds =  np.vstack(trainer.predict(model, valid_loader))[:,0 ,: ,: ]
np.save(PRED_NPY, preds)

if save_truth:
    true_masks = np.vstack([valid_dataset[i]['mask'] for i in range(len(valid_dataset))])
    np.save(TRUE_NPY, true_masks)
