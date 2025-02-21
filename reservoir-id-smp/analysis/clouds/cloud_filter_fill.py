import rasterio as rio
from rasterio.enums import Resampling
from rasterio.windows import Window
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter


BOX_SIZE = 5000 # CLOUD box size. Tripled for res

def read_cloud(year, box_size_rows, box_size_cols, row, col, upscale_factor=3):
    row_end = not (box_size_rows==BOX_SIZE)
    col_end = not (box_size_cols==BOX_SIZE)
    row_start = (row==0)
    col_start = (col==0)
    cloud_path = './data/ls5_{}_clouds.tif'.format(year)
    if not os.path.isfile(cloud_path):
        return np.ones((box_size_cols*upscale_factor, box_size_rows*upscale_factor),
                       dtype=np.uint8)*255
    else:
        fh_cloud = rio.open(cloud_path)
        padded_box_size_rows = box_size_rows + (4*(not row_end))+(4*(not row_start))
        padded_box_size_cols = box_size_cols + (4*(not col_end))+(4*(not col_start))
        cloud_ar = fh_cloud.read(
            1,
            window=Window(np.max(col-4,0), np.max(row-4,0), 
                          padded_box_size_cols, padded_box_size_rows
                          ),
            out_shape = (
                int((padded_box_size_cols) * upscale_factor),
                int((padded_box_size_rows)* upscale_factor)
                ), 
            resampling=Resampling.average
            ) 
        if row_end:
            row_clip = None
        else:
            row_clip = -12
        if col_end:
            col_clip = None
        else:
            col_clip = -12

        return maximum_filter(cloud_ar, size=10)[
            ((not col_start)*12):col_clip, ((not row_start)*12):row_clip]

def read_res(year, box_size_rows, box_size_cols, row, col, upscale_factor=3):
    res_path = './data/preds/ls5_{}_v3.tif'.format(year)
    if not os.path.isfile(res_path):
        return np.ones((box_size_cols*upscale_factor, box_size_rows*upscale_factor),
                       dtype=np.uint8)*255
    else:
        fh_res = rio.open(res_path)
        return fh_res.read(
            1,
            window=Window(col*upscale_factor,
                        row*upscale_factor,
                        box_size_cols*upscale_factor,
                        box_size_rows*upscale_factor)
            )

def create_stacks(y, box_size_rows, box_size_cols, row_off, col_off):
    priority = [2, 1, 3, 0, 4]
    cloud_stack = np.array([read_cloud(y, box_size_rows, box_size_cols, row_off, col_off)
                            for y in range(y-2,
                                            y+3)])[priority]
    
    res_stack = np.array([read_res(y, box_size_rows, box_size_cols, row_off, col_off)
                            for y in range(y-2,
                                            y+3)])[priority]
    return res_stack, cloud_stack

def update_stacks(res_stack, cloud_stack, y, box_size_rows, box_size_cols, row_off, col_off):
    reshuffle_order = np.array([2, 0, 4, 1])
    cloud_stack = cloud_stack[reshuffle_order]
    res_stack = res_stack[reshuffle_order]
    cloud_stack = np.append(cloud_stack, [read_cloud(y+2, box_size_rows, box_size_cols, row_off, col_off)], axis=0)
    res_stack = np.append(res_stack, [read_res(y+2, box_size_rows, box_size_cols, row_off, col_off)], axis=0)
    return res_stack, cloud_stack


def fill_vals(res_stack, cloud_stack, cloud_cutoff=75):
    cloud_cutoff=75
    res_out = res_stack[0].copy()
    cloud_stack[res_stack==255] = 255
    mask = (cloud_stack[0] > cloud_cutoff)
    # best_clouds = np.argmin(cloud_stack[:,mask], axis=0)
    best_clouds = np.argmin(cloud_stack[:,mask]>cloud_cutoff, axis=0)
    replacement_vals = res_stack[best_clouds, mask]
    replacement_vals[best_clouds==0] = 255
    # replacement_vals[cloud_stack[best_clouds, mask] > cloud_cutoff] = 255
    res_out[mask] = replacement_vals
    return res_out

# def write_res(res_out, res_stack, cloud_stack, y, row_off, col_off, box_size_rows,box_size_cols,
def write_res(res_out, y, row_off, col_off, box_size_rows,box_size_cols,
              old_transform, out_profile):
    affine_transformer = rio.transform.AffineTransformer(old_transform)
    out_profile['height'] = box_size_rows*3
    out_profile['width'] = box_size_cols*3
    new_x, new_y = affine_transformer.xy(row_off*3-0.5, col_off*3-0.5)

    new_affine = rio.transform.Affine(
            a=old_transform.a,
            b=old_transform.b,
            c=new_x,
            d=old_transform.d,
            e=old_transform.e,
            f=new_y
            )
    out_profile['transform'] = new_affine

    # out_clouds = './out/test_clouds_{}_{}_{}.tif'.format(y, row_off, col_off)
    # out_res_old = './out/test_raw_{}_{}_{}.tif'.format(y, row_off, col_off)
    out_res_new = './out/test_filt_{}_{}_{}.tif'.format(y, row_off, col_off)
    # with rio.open(out_clouds, 'w', **out_profile) as dst:
    #     dst.write(cloud_stack[0], indexes=1)
    # with rio.open(out_res_old, 'w', **out_profile) as dst:
    #     dst.write(res_stack[0], indexes=1)
    with rio.open(out_res_new, 'w', **out_profile) as dst:
        dst.write(res_out, indexes=1)

def fully_fill(res_stack, cloud_stack, y, row_off, col_off, box_size_rows, box_size_cols,
               old_transform, out_profile):
    if res_stack is None:
        res_stack, cloud_stack = create_stacks(y, box_size_rows, box_size_cols, row_off, col_off)
    else:
        res_stack, cloud_stack = update_stacks(res_stack, cloud_stack, y, box_size_rows, box_size_cols, row_off, col_off)
    
    res_out = fill_vals(res_stack, cloud_stack)

    # write_res(res_out, res_stack, cloud_stack, y, row_off, col_off, box_size_rows, box_size_cols,
    write_res(res_out, y, row_off, col_off, box_size_rows, box_size_cols,
              old_transform,out_profile)

    return res_stack, cloud_stack


fh_res = rio.open('./data/preds/ls5_1984_v3.tif')
fh_clouds = rio.open('./data/ls5_1984_clouds.tif')
old_transform = fh_res.transform
out_profile = fh_res.profile.copy()
total_rows, total_cols = fh_clouds.height, fh_clouds.width
current_row = 0
current_col = 0
row_starts = np.arange(0, total_rows, BOX_SIZE)
col_starts = np.arange(0, total_cols, BOX_SIZE)

# Create Nx2 array with row/col start indices
start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

# Get some affine information
cur_transform = fh_res.transform
affine_transformer = rio.transform.AffineTransformer(cur_transform)

for i in range(start_ind.shape[0]):
    res_stack=None
    cloud_stack=None
    # For the indices near edge we need to use a smaller box size
    box_size_rows = min(total_rows - start_ind[i,0], BOX_SIZE)
    box_size_cols = min(total_cols - start_ind[i,1], BOX_SIZE)
    # Get base counts and labeled image
    start_ind_row, start_ind_col = (start_ind[i,0], start_ind[i,1])
    for y in range(1984, 1990):
        res_stack, cloud_stack = fully_fill(res_stack, cloud_stack, y, start_ind_row, start_ind_col,
                                            box_size_rows, box_size_cols,
                                            cur_transform, fh_res.profile)
