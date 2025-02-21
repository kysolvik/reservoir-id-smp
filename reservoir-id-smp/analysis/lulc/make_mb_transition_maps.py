import rasterio as rio
import pandas as pd
import numpy as np
import numba as nb

mb_keys_dict = {
    'crop': np.array([18,19,39,20,40,62,41,36,46,47,35,48]),
    'forest': np.array([3]),
    'savanna': np.array([4]),
    'grassland':np.array([12]),
    'pasture': np.array([15])
}

transition_keys = []
for from_cover in mb_keys_dict.keys():
    for to_cover in mb_keys_dict.keys():
        transition_keys.append('{}-{}'.format(from_cover, to_cover))

transition_dict = {}
for val, transition in enumerate(transition_keys):
    transition_dict[transition] = val+1

@nb.njit(parallel=True)
def nb_isin_listcomp(matrix, index_to_remove):
    #matrix and index_to_remove have to be numpy arrays
    #if index_to_remove is a list with different dtypes this 
    #function will fail
    og_shape = matrix.shape
    matrix = matrix.flatten()
    out=np.empty(matrix.shape[0],dtype=nb.boolean)
    index_to_remove_set=set(index_to_remove)

    for i in nb.prange(matrix.shape[0]):
        if matrix[i] in index_to_remove_set:
            out[i]=True
        else:
            out[i]=False

    return out.reshape(og_shape)


def calc_mask(ar, cover):
    if cover == 'crop':
        mask = nb_isin_listcomp(ar, mb_keys_dict['crop'])
    else:
        mask = ar==mb_keys_dict[cover][0]
    return mask

def make_transition_maps(y):
    og_fp =  './in/mato_grosso/aea/mapbiomas-brazil-collection-90-matogrossomt-{}_aea_30m.tif'.format(y)
    og_ar = rio.open(og_fp).read()

    compare_y = y+1
    new_fp = './in/mato_grosso/aea/mapbiomas-brazil-collection-90-matogrossomt-{}_aea_30m.tif'.format(compare_y)
    new_ar = rio.open(new_fp).read()

    # Write out values
    out_ar = np.zeros_like(new_ar)
    for from_type in mb_keys_dict.keys():
        for to_type in mb_keys_dict.keys():
            combined_mask = calc_mask(og_ar, from_type)*calc_mask(new_ar, to_type)
            out_ar[combined_mask] = transition_dict['{}-{}'.format(from_type, to_type)]
            
    return out_ar

src = rio.open('./in/mato_grosso/aea/mapbiomas-brazil-collection-90-matogrossomt-1985_aea_30m.tif')
def write_raster(ar, out_path, src):
    with rio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src.profile

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rio.uint8,
            count=1,
            compress='lzw')

        with rio.open(out_path, 'w', **profile) as dst:
            dst.write(ar[0].astype(rio.uint8), 1)


def main():
    for y in range(1985, 2023):
        trans_ar = make_transition_maps(y)

        out_path = './out/lulc_transition_maps/mb_transition_{}_{}.tif'.format(y, y+1)
        write_raster(trans_ar, out_path,src)

if __name__=='__main__':
    main() 
