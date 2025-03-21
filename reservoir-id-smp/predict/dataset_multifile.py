from torch.utils.data import Dataset as BaseDataset
from torch.utils.data.dataloader import default_collate
import torch
import os
from skimage import io
import numpy as np
import albumentations as albu
import affine
import time
import rasterio


class ResDatasetMultiFile(BaseDataset):
    """

    Args:
        fhs (rasterio filehandles):
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)

    """


    def __init__(
            self,
            start_inds,
            fhs,
            bands_minmax,
            band_selection,
            tile_rows,
            tile_cols,
            overlap,
            out_dir,
            add_nds,
            mean_std=None,
    ):
        self.ids = np.arange(start_inds.shape[0])
        self.start_inds = start_inds
        self.fhs = fhs
        self.band_selection = band_selection
        self.overlap = overlap
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.out_dir = out_dir
        self.bands_minmax = bands_minmax
        self.src_transform = self.fhs[0].transform
        self.preprocessing_to_tensor = self.get_preprocessing_to_tensor()
        self.mean_std = mean_std
        self.add_nds = add_nds


    def to_tensor(self, x, **kwargs):
        """Tensor needs reordered indices"""
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing_to_tensor(self):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            albu.Lambda(image=self.to_tensor, mask=self.to_tensor),
        ]
        return albu.Compose(_transform)

    def get_geotransform(self, indice_pair):
        """Calculate geotransform of a tile.

        Notes:
            Using .affine instead of .transform because it should work with all
            rasterio > 0.9. See https://github.com/mapbox/rasterio/issues/86.

        Args:
            indice_pair (tuple): Row, Col indices of upper left corner of tile.
            src_transform (tuple): Geo transform/affine of src image

        """
        if self.overlap > 0:
            indice_pair = (indice_pair[0]+(self.overlap/2),
                           indice_pair[1]+(self.overlap/2))
        new_ul = [self.src_transform[2] + indice_pair[0]*self.src_transform[0] + indice_pair[1]*self.src_transform[1],
                  self.src_transform[5] + indice_pair[0]*self.src_transform[3] + indice_pair[1]*self.src_transform[4]]

        new_affine = affine.Affine(self.src_transform[0], self.src_transform[1], new_ul[0],
                                   self.src_transform[3], self.src_transform[4], new_ul[1])

        return new_affine

    def __getitem__(self, i):

        # read data
        image = self.load_single_image(self.start_inds[i])
        image = self.preprocess(image, self.bands_minmax, self.mean_std, self.band_selection)

        # if self.mean_std is not None:
        #    image = self.normalize_image(image, self.mean_std)

        # apply preprocessing
        if self.preprocessing_to_tensor:
            sample = self.preprocessing_to_tensor(image=image)
            image = sample['image']

        # Get geo transform information
        geo_transform = self.get_geotransform((self.start_inds[i,0], self.start_inds[i,1]))

        # Get output filename
        outfile = '{}/pred_{}-{}.tif'.format(
                self.out_dir, self.start_inds[i, 0], self.start_inds[i, 1])



        return {'image': image,
                'geo_transform': geo_transform,
                'outfile': outfile,
                }

    def __len__(self):
        return len(self.ids)

    def normalized_diff(self, ar1, ar2):
        """Returns normalized difference of two arrays."""

        return np.nan_to_num(((ar1 - ar2) / (ar1 + ar2)), 0)

    def calc_all_nds(self, img):

        nd_list =[]

        # SENTINEL ND bands
        # Add  Gao NDWI
        nd_list += [self.calc_nd(img, 3, 7)]
        # Add  MNDWI
        nd_list += [self.calc_nd(img, 1, 7)]
        # Add McFeeters NDWI band
        nd_list += [self.calc_nd(img, 1, 3)]
        # Add NDVI band
        nd_list += [self.calc_nd(img, 3, 2)]
        
        return np.stack(nd_list, axis=2)

    def calc_nd(self, img, band1, band2):
        """Add band containing NDWI.. Slightly different for LS and sentinel (dims)"""

        nd = self.normalized_diff(img[:,:,band1].astype('float64'),
                                  img[:,:,band2].astype('float64'))

        # Rescale to uint8
        nd = np.round(255.*(nd - (-1))/(1 - (-1)))
        if nd.max() > 255:
            print(nd.max())
            print('Error: overflow')

        return nd.astype(np.uint8)

    def rescale_to_minmax_uint8(self, img, bands_minmax):
        """Rescales images to 0-255 based on (precalculated) min/maxes of bands"""
        img = np.where(img > bands_minmax[1], bands_minmax[1], img)
        img = (255. * (img.astype('float64') - bands_minmax[0]) / (bands_minmax[1] - bands_minmax[0]))
        img = np.round(img)
        if img.max() > 255:
            print(img.max())
            print('Error: overflow')
        return img.astype(np.uint8)

#    def normalize_image(self, ar, mean_std):
#        return (ar - mean_std[0])/mean_std[1]

    def preprocess(self, img, bands_minmax, mean_std, band_selection):
        """Prep the input images"""
        img = self.rescale_to_minmax_uint8(img, bands_minmax)
        if self.add_nds:
            nds = self.calc_all_nds(img)
            img = np.concatenate([img, nds], axis=2)[:, :, band_selection]
        else:
            img = img[:,:, band_selection]
        # For sentinel, mean_std is already the same size
        if mean_std.shape[1] == len(band_selection):
            img = (img - mean_std[0])/mean_std[1]
        else:
            img = (img - mean_std[0, band_selection])/mean_std[1, band_selection]

        return img

    def collate(self, batch):
        batch_out = {}
        for key in batch[0]:
            batch_out[key] = [elem[key] for elem in batch]

        batch_out['image'] = default_collate(batch_out['image'])
        return batch_out


    def load_single_image(self, start_inds):
        """Load single tile from list of src"""
        col, row = start_inds[0], start_inds[1]

        # Try/except
        try:
            og_img_list = [f.read(window=((row, row + self.tile_rows),
                                          (col, col + self.tile_cols))) for f in self.fhs]
        except:
            og_img_list = [f.read(window=((row, row + self.tile_rows),
                                          (col, col + self.tile_cols))) for f in self.fhs]

        return np.moveaxis(np.vstack(og_img_list), 0, -1)
