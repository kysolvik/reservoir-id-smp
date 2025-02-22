#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import PIL
import PIL.Image
import glob
from skimage import io
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
import pandas as pd
import subprocess as sp


# In[2]:


random.seed(171) # 205 for last round of SR


# In[3]:


input_dir = './data/landsat8_10m_2017_sr_720/'
out_dir = './data/landsat8_v10/'


# In[4]:


sp.call(['mkdir', '-p', out_dir + 'img_dir/train'])
sp.call(['mkdir', '-p', out_dir + 'img_dir/val'])
sp.call(['mkdir', '-p', out_dir + 'img_dir/test'])

sp.call(['mkdir', '-p', out_dir + 'ann_dir/train'])
sp.call(['mkdir', '-p', out_dir + 'ann_dir/val'])
sp.call(['mkdir', '-p', out_dir + 'ann_dir/test'])


# In[5]:


val_frac = 0.20
test_frac = 0.20


# In[6]:


remove_cloudy_images = True
cloudy_csv = './data/cloud_images.csv'
replace_bad_masks = True
bad_mask_csv = './data/replace_w_zeromask.csv'


# In[7]:


if remove_cloudy_images:
    cloudy_base_list = pd.read_csv(cloudy_csv)['name'].values
else:
    cloudy_base_list = np.array([])
if replace_bad_masks:
    replace_mask_base_list = pd.read_csv(bad_mask_csv)['name'].values
else:
    replace_mask_base_list = np.array([])


# ## First round of loading data:
#  1. Save mins and maxes of every band for every image

# In[8]:


# Get mask image names and base image patterns
mask_images = glob.glob('{}*mask.png'.format(input_dir))
mask_images.sort()
image_patterns = [mi.replace('mask.png', '') for mi in mask_images]

band_mins = []
band_maxes = []

bad_file_list = []
zero_file_list = []
                  
for image_base in image_patterns:
    try:
        stacked_ar = io.imread('{}ls8_10m.tif'.format(image_base))
    except:
        print('Bad file: {}'.format(image_base))
        bad_file_list.append(os.path.basename('{}ls8_10m.tif'.format(image_base)))
        pass
        continue
                               
    
    img_min = np.min(stacked_ar, axis=(0,1))
    img_max = np.max(stacked_ar, axis=(0,1))
    
    if np.min(img_min)==0 and np.max(img_max)==0:
        print('Zero img: {}'.format(image_base))
        zero_file_list.append(os.path.basename('{}ls8_10m.tif'.format(image_base)))
        continue
    
    
    band_mins += [img_min]
    band_maxes += [img_max]

all_mins = np.stack(band_mins)
all_maxes = np.stack(band_maxes)
bands_min_max_all_imgs = np.stack([all_mins, all_maxes], axis=0)
np.save('./data/bands_minmax/landsat_all_imgs_bands_min_max_ls8_10m_v10.npy', bands_min_max_all_imgs)


# In[9]:


# for f in zero_file_list:
#     full_path = input_dir + f
#     os.remove(full_path)
# for f in bad_file_list:
#     full_path = input_dir + f
#     os.remove(full_path)


# ## Second round of loading data:
# 1. Skip any cloudy images
# 2. Split into train, val, test
# 3. Rescale bands using the previously calculated min and max
# 4. Calc NDs
# 5. Select 3 bands for feeding into CNN
# 6. Calc mean and std in running fashion for those bands, print it out
# 

# In[10]:


bands_min_max_all_imgs = np.load('./data/bands_minmax/landsat_all_imgs_bands_min_max_ls8_10m_v10.npy')
bands_min_max = np.array([np.min(bands_min_max_all_imgs[0], axis=0),
                          np.percentile(bands_min_max_all_imgs[1], 80, axis=0)])
# bands_min_max = np.array([np.min(bands_min_max_all_imgs[0], axis=0),
#                           np.percentile(bands_min_max_all_imgs[1], 90, axis=0)])


# In[11]:


def normalized_diff(ar1, ar2):
    """Returns normalized difference of two arrays."""
    
    return np.nan_to_num(((ar1 - ar2) / (ar1 + ar2)),0)
    
    
def calc_nd(img, band1, band2):
    """Add band containing NDWI."""

    nd = normalized_diff(img[:,:,band1].astype('float64'), 
                         img[:,:,band2].astype('float64'))
    
    # Rescale to uint8
    nd = np.round(255.*(nd - (-1))/(1 - (-1)))
    if nd.max()>255:
        print(nd.max())
        print('Error: overflow')
   
    return nd.astype(np.uint8)

def calc_all_nds(img):
    
    nd_list =[]
    
    # Add  Gao NDWI
    nd_list += [calc_nd(img, 4, 5)]
    # Add  MNDWI
    nd_list += [calc_nd(img, 2, 5)]
    # Add McFeeters NDWI band
    nd_list += [calc_nd(img, 2, 4)]
    # Add NDVI band
    nd_list += [calc_nd(img, 4, 3)]

    return np.stack(nd_list, axis=2)


# In[12]:


def rescale_to_minmax_uint8(img, bands_min_max):
    img = np.where(img > bands_min_max[1], bands_min_max[1], img)
    img  = (255. * (img.astype('float64') - bands_min_max[0]) / (bands_min_max[1] - bands_min_max[0]))
    img = np.round(img)
    if img.max()>255:
        print(img.max())
        print('Error: overflow')
    return img.astype(np.uint8)

def select_bands_save_images(fp_base, out_path, band_selection, bands_min_max, crop_size, resample_size,
                          calc_mean_std=False):
    ar = io.imread('{}ls8_10m.tif'.format(fp_base))
    
    ar = rescale_to_minmax_uint8(ar, bands_min_max)
    
    nds = calc_all_nds(ar)
    
    ar = np.concatenate([ar, nds], axis=2)[:, :, band_selection]
    
    if ar.shape[:-1] != crop_size:
        ar = crop(ar, crop_size)
    if ar.shape[:-1] != resample_size:
        ar = resize(ar, resample_size, preserve_range=True).astype(np.uint8)
    io.imsave(out_path, ar)
    if calc_mean_std:
        return ar.reshape((-1, len(band_selection)))
    
def crop(img, out_size):
    crop_size = int((img.shape[0] - out_size[0])/2)
    img = img
    return img[crop_size:(-crop_size), crop_size:(-crop_size)]

def create_nband_images(img_list, output_dir, band_selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], crop_size=(256,256),
                      resample_size=(256, 256),
                      calc_mean_std=False):
    if calc_mean_std:
        n = 0
        mean = np.zeros(len(band_selection))
        sums =  np.zeros(len(band_selection))
        M2 =  np.zeros(len(band_selection))

    for fp_base in img_list:
        # out_path is a little tricky, need to remove _ at end and add in .tif
        out_path = fp_base.replace(input_dir, output_dir)[:-1] + '.tif'
        if calc_mean_std:
            vals = select_bands_save_images(fp_base, out_path, band_selection, 
                                         bands_min_max, crop_size, resample_size, 
                                         calc_mean_std=calc_mean_std)
            n += vals.shape[0]
            vals = vals
            sums += np.sum(vals, axis=0)
            delta = vals - mean
            mean += np.sum(delta/n, axis=0)
            M2 += np.sum(delta*(vals - mean), axis=0)
        else:
            select_bands_save_images(fp_base, out_path, band_selection, bands_min_max, crop_size, resample_size)
            
    if calc_mean_std:
        return sums/n, np.sqrt(M2 / (n - 1))
    
        
def save_mask_images(ann_list, output_dir, resample_size=(166, 166)):
    total_res_pixels = 0
    for fp_base in ann_list:
        fp = '{}mask.png'.format(fp_base)
        ar = io.imread(fp)
        ar[ar>0] = 1
        
        # Sometimes the imagess have 3 dims. We only want 2
        if len(ar.shape) > 2:
            ar = ar[:,:,0]
            
        # Replace bad masks
        if os.path.basename(fp).replace('_mask.png', '') in replace_mask_base_list:
            ar[:] = 0
        
        # Replace if only 1 positive pixel
        ar_sum = ar.sum()
        if ar_sum == 1:
            ar[:] = 0
        else:
            total_res_pixels += ar_sum
            
        # Resize
        if ar.shape != resample_size:
            ar = np.round(resize(ar, resample_size, preserve_range=True)).astype(np.uint8)

        if np.sum(np.logical_and(ar!=1, ar!=0)) > 0:
            print(fp)
            print(np.unique(ar))
            raise ValueError('Mask has non-0 and/or non-1 values')
            
        # Save
        out_path = fp.replace(input_dir, output_dir).replace('_mask.png', '.tif')
        io.imsave(out_path, ar)
        
    return total_res_pixels


# In[13]:


def split_train_test(img_patterns, test_frac, val_frac):
    """Split data into train, test, val (or just train)

    Returns:
        train_indices, val_indices, test_indices tuple
    """
    total_ims = len(img_patterns)
    if test_frac != 0:

        train_count = round(total_ims * (1 - test_frac - val_frac))
        train_indices = random.sample(range(total_ims), train_count)
        test_val_indices = np.delete(np.array(range(total_ims)), train_indices)

        test_count = round(total_ims * test_frac)
        test_indices = random.sample(list(test_val_indices), test_count)


        if val_frac != 0:
            val_indices = np.delete(np.array(range(total_ims)),
                                    np.append(train_indices, test_indices))

            return train_indices, val_indices, test_indices
        else: 
            return train_indices, test_indices
    else:
        return np.arange(total_ims)

# Get train, test, and val lists, skipping cloudy images
def list_and_split_imgs(input_dir, cloudy_base_list):
    # First get list of images
    mask_images = glob.glob('{}*mask.png'.format(input_dir))
    mask_images.sort()
    image_patterns = [mi.replace('mask.png', '') for mi in mask_images if os.path.isfile(mi.replace('mask.png', 'ls8_10m.tif'))]
    image_patterns = np.array([image_pat for image_pat in image_patterns if not os.path.basename(image_pat) in cloudy_base_list])

    # No floodplain images in test set
    floodplain_images = np.array([ip for ip in image_patterns if 'floodplains' in ip])
    non_floodplain_images = np.array([ip for ip in image_patterns if 'floodplains' not in ip])

    # Separate zero masks from others
    mask_sums = np.array([np.sum(io.imread(ip + 'mask.png')) for ip in non_floodplain_images])
    zero_mask_images = non_floodplain_images[mask_sums==0]
    res_mask_images = non_floodplain_images[mask_sums!=0]
    train_indices_zero, val_indices_zero, test_indices_zero = split_train_test(zero_mask_images, test_frac=test_frac, val_frac=val_frac)
    train_indices_res, val_indices_res, test_indices_res = split_train_test(res_mask_images, test_frac=test_frac, val_frac=val_frac)
    # train_indices = np.concatenate([train_indices_zero, train_indices_res])
    # val_indices = np.concatenate([val_indices_zero, val_indices_res])
    # test_indices = np.concatenate([test_indices_zero, test_indices_res])

    # If not separating zero masks
    # train_indices, val_indices, test_indices = split_train_test(non_floodplain_images, test_frac=test_frac, val_frac=val_frac)

    # For including FP in val Using test_frac arg because allows for no 3rd split
    # train_indices_fp, val_indices_fp = split_train_test(floodplain_images, test_frac=val_frac, val_frac=0)
    # train_basename_list = np.concatenate([non_floodplain_images[train_indices], floodplain_images[train_indices_fp]])
    # val_basename_list = np.concatenate([non_floodplain_images[val_indices], floodplain_images[val_indices_fp]])
    # test_basename_list = non_floodplain_images[test_indices]

    # # For only putting floodplains in train
    # train_basename_list = np.concatenate([non_floodplain_images[train_indices], floodplain_images])
    # val_basename_list = non_floodplain_images[val_indices]
    # test_basename_list = non_floodplain_images[test_indices]

    # For stratified sampling
    train_basename_list = np.concatenate([zero_mask_images[train_indices_zero], res_mask_images[train_indices_res], floodplain_images])
    val_basename_list = np.concatenate([zero_mask_images[val_indices_zero], res_mask_images[val_indices_res]])
    test_basename_list = np.concatenate([zero_mask_images[test_indices_zero], res_mask_images[test_indices_res]])
    
    return train_basename_list, val_basename_list, test_basename_list


# In[14]:


train_basename_list, val_basename_list, test_basename_list = list_and_split_imgs(input_dir, cloudy_base_list)


# In[15]:


# For 10m Landsat
means_std = create_nband_images(train_basename_list, output_dir = '{}/img_dir/train/'.format(out_dir),
                              calc_mean_std = True, resample_size=(720,720), crop_size=(720,720))
create_nband_images(test_basename_list, output_dir = '{}/img_dir/test/'.format(out_dir), calc_mean_std = False,
                   resample_size=(720,720), crop_size=(720,720))
create_nband_images(val_basename_list, output_dir = '{}/img_dir/val/'.format(out_dir), calc_mean_std = False,
                   resample_size=(720,720), crop_size=(720,720))


# In[16]:


# For 10m Landsat
sum_res_pixels = save_mask_images(train_basename_list, output_dir = '{}/ann_dir/train/'.format(out_dir),
                                 resample_size=(500, 500))
save_mask_images(test_basename_list, output_dir = '{}/ann_dir/test/'.format(out_dir),
                                 resample_size=(500, 500))
save_mask_images(val_basename_list, output_dir = '{}/ann_dir/val/'.format(out_dir),
                                 resample_size=(500, 500))


# In[17]:


np.save('./data/mean_stds/mean_std_ls8_v10.npy', np.vstack(means_std))

