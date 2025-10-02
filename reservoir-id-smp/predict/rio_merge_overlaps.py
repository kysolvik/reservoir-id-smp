import numpy as np
import rasterio as rio
import glob
from rasterio.merge import merge
import os
import subprocess as sp
from rasterio.windows import Window


PROCESS_BOX_SIZE = 20000
IN_DIR = './out_allbrazil_v12_30m/sentinel/2021'
OUT_DIR = './merge_test3'

os.makedirs(OUT_DIR)


def get_files(input_dir):
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.tif')))
    return input_files

def template_vrt_info(file_list, make_vrt=True):
    if make_vrt:
        with open('./temp_filelist.txt', 'w') as f:
            for fp in file_list:
                f.write(fp + '\n')
                
        sp.call(['gdalbuildvrt', 'temp.vrt', '-input_file_list', './temp_filelist.txt'])
        os.remove('./temp_filelist.txt')
    return rio.open('./temp.vrt')

def filter_bounds(tile_bounds, rows, cols):
    """
    Args:
        tile_bounds (list): start_row, start_col, end_row, end_col
    """

    file_inds = ((rows > tile_bounds[0])
                 * (rows < tile_bounds[2])
                 * (cols > tile_bounds[1])
                 * (cols < tile_bounds[3]))
    return file_inds

def row_col_start(fp):
    row = fp.split('_')[-1].split('-')[0]
    col = fp[:-4 ].split('-')[-1]
    return np.array([int(row), int(col)])

# def open_src(src_files):
#     return [rio.open(src) for src in src_files]

def merge_avg(src_files, geo_bounds):
    # file_handles = []
    # file_handles = [rio.open(src) for src in src_files]
    # for src in src_files:
    #     file_handles.append(rio.open(src))
    # with open_src(src_files) as file_handles:
    # print(file_handles)
    # Issues with overflow in the sum
    sum, out_trans = merge(src_files, method='sum', bounds=geo_bounds, dtype=np.float64)
    count, out_trans = merge(src_files, method='count', bounds=geo_bounds)
    avg = sum/count
    avg[count==255] = 255
    # for fh in file_handles:
    #     fh.close()
    return avg, out_trans

def create_output(out_path, input_vrt):
    out_profile = input_vrt.profile
    out_profile['driver'] = 'GTiff'
    out_profile['compress'] = 'LZW'
    out_profile['blockxsize'] = 256
    out_profile['blockysize'] = 256
    out_fh = rio.open(
        out_path,
        'w',
        **out_profile
        )
    return out_fh
    # out_fh.close()


def write_out(avg_ar, out_path, out_fh, row_start, col_start):
    print(avg_ar.shape)
    print(row_start)
    print(col_start)
    out_window = Window(col_off=col_start, row_off=row_start,
                        width=avg_ar.shape[1], height=avg_ar.shape[0])
    print(out_window)
    out_fh.write(avg_ar, window=out_window, indexes=1)
    

# Get input files
file_list = np.array(get_files(IN_DIR))
tile_size = rio.open(file_list[0]).width
rows_cols = np.array([row_col_start(fp) for fp in file_list])

# Make a template vrt and set up output profile using it
vrt_info = template_vrt_info(file_list)
total_rows, total_cols = vrt_info.shape

# Build output raster profile
out_profile = vrt_info.profile
out_profile['driver'] = 'GTiff'
out_profile['compress'] = 'LZW'
out_profile['blockxsize'] = 256
out_profile['blockysize'] = 256

# Process raster in blocks, each PROCESS_BOX SIZE x PROCESS_BOX_SIZE
row_starts = np.arange(0, total_rows, PROCESS_BOX_SIZE)
col_starts = np.arange(0, total_cols, PROCESS_BOX_SIZE)

start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)
for cur_ind in start_ind:
    # Merge can't handle 300,000 files, so first we filter by indexes
    tile_bounds = [cur_ind[0] - tile_size,
                    cur_ind[1] - tile_size,
                    cur_ind[0] + PROCESS_BOX_SIZE + tile_size,
                    cur_ind[1] + PROCESS_BOX_SIZE + tile_size]
    target_files = file_list[filter_bounds(tile_bounds, rows_cols[:, 0], rows_cols[:, 1])]
    if len(target_files)> 0:
        # Get geographic bounds info
        geo_starts = vrt_info.transform * cur_ind
        geo_ends = list(vrt_info.transform * (cur_ind + PROCESS_BOX_SIZE))
        geo_ends[1] = max(geo_ends[1], vrt_info.bounds.bottom)
        geo_ends[0] = min(geo_ends[0], vrt_info.bounds.right)
        geo_bounds = (geo_starts[0], geo_ends[1], geo_ends[0], geo_starts[1])

        avg, out_transform = merge_avg(target_files, geo_bounds)
        if geo_starts[0] != out_transform.c or geo_starts[1] != out_transform.f:
            print('geo bounds:', geo_bounds)
            print('out_transform:', out_transform)
            raise ValueError('Output transform does not match geo bounds')

        # avg is shape (1, height, width)
        out_profile['height'] = avg.shape[1]
        out_profile['width'] = avg.shape[2]
        out_profile['transform'] = out_transform
        with rio.open(
            os.path.join(OUT_DIR, 'overlapped_{}_{}.tif'.format(cur_ind[0], cur_ind[1])),
            'w',
            **out_profile) as out_fh:
            out_fh.write(avg.astype(np.uint8))

# Build output vrt and translate
sp.call('gdalbuildvrt',
        '-co', 'BLOCKXSIZE=256',
        '-co', 'BLOCKXSIZE=256',
        os.path.join(OUT_DIR, 'full_overlapped.vrt'),
        os.path.join(OUT_DIR, 'overlapped*.tif')
        )
sp.call('gdal_translate',
        '-co', 'COMPRESS=LZW',
        '-co', 'TILED=YES',
        '-co', 'BLOCKXSIZE=256',
        '-co', 'BLOCKYSIZE=256',
        os.path.join(OUT_DIR, 'full_overlapped.vrt'),
        os.path.join(OUT_DIR, 'full_overlapped.tif')
        )

# NOTE: Windowed writing to a big file worked for a while but hung after about 8 windows
# So taking the more annoying but more straightforward route of writing tiles
# with create_output('./merge_test.tif', vrt_info) as out_fh:
#     for cur_ind in start_ind[12:]:
#         tile_bounds = [cur_ind[0] - 1000, cur_ind[1] - 1000, cur_ind[0] + box_size + 1000, cur_ind[1] + box_size + 1000]
#         target_files = file_list[filter_bounds(tile_bounds, rows_cols[:, 0], rows_cols[:, 1])]
#         if len(target_files)> 0:
#             print(target_files[0])
#             geo_starts = vrt_info.transform * cur_ind
#             geo_ends = list(vrt_info.transform * (cur_ind + box_size))
#             geo_ends[1] = max(geo_ends[1], vrt_info.bounds.bottom)
#             geo_ends[0] = min(geo_ends[0], vrt_info.bounds.right)

#             geo_bounds = (geo_starts[0], geo_ends[1], geo_ends[0], geo_starts[1])
#             print(geo_bounds)
#             avg, out_trans = merge_avg(target_files, geo_bounds)
#             print(avg.shape)
#             print(np.max(avg.astype(np.uint8)))
#             write_out(avg.astype(np.uint8)[0], out_fh, cur_ind[1], cur_ind[0])

