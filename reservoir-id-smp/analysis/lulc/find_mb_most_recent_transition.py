import rasterio as rio
import pandas as pd
import numpy as np
import numba as nb

MB_KEYS_DICT = {
    'crop': np.array([18,19,39,20,40,62,41,36,46,47,35,48]),
    'forest': np.array([3]),
    'savanna': np.array([4]),
    'grassland':np.array([12]),
    'pasture': np.array([15]),
    'mosaic':np.array([21])
}

COVER_DICT = {}
for val, transition in enumerate(MB_KEYS_DICT.keys()):
    COVER_DICT[transition] = val+1
COVER_DICT['other'] = 0


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
        mask = nb_isin_listcomp(ar, MB_KEYS_DICT['crop'])
    else:
        mask = ar==MB_KEYS_DICT[cover][0]
    return mask

def get_combined_mask(y):
    new_fp =  './in/mato_grosso/aea/mapbiomas-brazil-collection-90-matogrossomt-{}_aea_30m.tif'.format(y)
    new_ar = rio.open(new_fp).read(1)

    # Write out values
    out_ar = np.zeros_like(new_ar)
    for cover_type in MB_KEYS_DICT.keys():
        cover_mask = calc_mask(new_ar, cover_type)
        out_ar[cover_mask] = COVER_DICT[cover_type]

    return out_ar



def main():
    cur_fh = rio.open('./in/mato_grosso/aea/mapbiomas-brazil-collection-90-matogrossomt-2023_aea_30m.tif')
    cur_ar = cur_fh.read(1)
    cur_pasture_mask = calc_mask(cur_ar, 'pasture')
    cur_crop_mask = calc_mask(cur_ar, 'crop')
    total_pasture = np.sum(cur_pasture_mask)
    total_crop = np.sum(cur_crop_mask)
    pasture_trans_dict = dict(
            zip(COVER_DICT.keys(),
                np.zeros(len(COVER_DICT.keys()))
                )
            )
    crop_trans_dict = dict(
            zip(COVER_DICT.keys(),
                np.zeros(len(COVER_DICT.keys()))
                )
            )
    for y in np.arange(2022, 1984, -1):
        print(y)
        combined_mask = get_combined_mask(y)

        for cover in pasture_trans_dict.keys():
            if cover != 'pasture':
                pasture_trans_dict[cover] += (
                        np.sum(combined_mask[cur_pasture_mask] == COVER_DICT[cover])
                        )
            if cover != 'crop':
                crop_trans_dict[cover] += (
                        np.sum(combined_mask[cur_crop_mask] == COVER_DICT[cover])
                        )

        cur_pasture_mask[combined_mask != COVER_DICT['pasture']] = 0
        cur_crop_mask[combined_mask != COVER_DICT['crop']] = 0

    pasture_trans_dict['pasture'] = np.sum(cur_pasture_mask)
    pasture_trans_dict['total_pasture'] = total_pasture
    crop_trans_dict['crop'] = np.sum(cur_crop_mask)
    crop_trans_dict['total_crop'] = total_crop
    pd.DataFrame([pasture_trans_dict, crop_trans_dict],
                 index=['pasture', 'crop']).to_csv('./out/most_recent_transitions.csv')

if __name__=='__main__':
    main() 
