import rasterio as rio
import pandas as pd
import numpy as np

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

output_list = []
for y in range(1985, 2023):
    out_dict = {'year': y}
    cur_tif = './out/lulc_transition_maps/mb_transition_{}_{}.tif'.format(y, y+1)
    cur_ar = rio.open(cur_tif).read(1)
    for lulc_key in transition_keys:
        out_dict[lulc_key] = np.sum(cur_ar==transition_dict[lulc_key])
    output_list.append(out_dict)
    print(y, 'done')

out_df = pd.DataFrame(output_list)
out_df.to_csv('./out/lulc_summaries/lulc_transitions_yearof_only.csv', index=False)

