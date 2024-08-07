import pytorch_lightning as pl
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torchvision
import rasterio
import affine

class ResModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, crs, **kwargs):
        super().__init__()
        self.model = smp.MAnet(encoder_name=encoder_name, in_channels=in_channels, classes=out_classes,
                               aux_params=dict(
                                   classes=1,
                                   dropout=None)
                               )
        self.crs = crs
        self.crop_transform = torchvision.transforms.CenterCrop(500)

    def write_imgs(self, preds, outfile, geo_transform):
        """Write a batch of predictions to tiffs"""

#         preds[preds >= 0.5] = 1
#         preds[preds < 0.5] = 0
        preds = np.round(preds*100).astype(np.uint8)
        for i in range(preds.shape[0]):
            new_dataset = rasterio.open(
                outfile[i], 'w', driver='GTiff',
                height=preds.shape[1], width=preds.shape[2],
                count=1, dtype='uint8', compress='lzw',
                crs=self.crs, nodata=0,
                transform=geo_transform[i]
            )
            pred = preds[i]
            new_dataset.write(pred.astype('uint8'), 1)

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

    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        preds = np.vstack(self(batch['image']).sigmoid())
        self.write_imgs(preds, batch['outfile'], batch['geo_transform'])# , batch['crs'])
        return 
