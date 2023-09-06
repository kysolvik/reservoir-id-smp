import pytorch_lightning as pl
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torchvision

class ResModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.MAnet(encoder_name=encoder_name, in_channels=in_channels, classes=out_classes,
                               aux_params=dict(
                                   classes=1,
                                   dropout=0.2)
                               )

        self.crop_transform = torchvision.transforms.CenterCrop(500)

    def forward(self, image):
        # normalize image here
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

        logits_mask = self.forward(image)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).byte()


        return pred_mask

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch['image']).sigmoid()
