import pytorch_lightning as pl
import torch
import numpy as np
import segmentation_models_pytorch as smp
import torchvision


class ResModel(pl.LightningModule):
    """Model class"""

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.Unet(encoder_name=encoder_name,
                              in_channels=in_channels,
                              classes=out_classes,
                              aux_params=dict(
                                classes=out_classes)
                              )
        self.crop_transform = torchvision.transforms.CenterCrop(500)

    def forward(self, image):
        mask = self.model(image)[0]
        return self.crop_transform(mask)

    def shared_step(self, batch, stage):
        image = batch["image"]

        assert image.ndim == 4

        # Check that image dimensions are divisible by 32
        # Otherwise will have error when passing through Neural Net
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        logits_mask = self.forward(image)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).byte()

        return pred_mask

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch['image']).sigmoid()
