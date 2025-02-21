# LULC Analysis

Linking reservoirs to LULC. There are several different analyses happening in this folder:


1. Overall LULC over the years

2. Mapping LULC transitions annually

3. Assigning reservoirs to LULC class

4. Connection reservoirs to LULC transition 

    - Method A:
        - Assign reservoirs to LULC class for all years (back and forward in time)
        - When the LULC class changes, count that reservoir as associated with that change +/- years between reservoir and LULC change.

    - Method B:
        - Same as Method A, but ONLY count closest LULC changes in time forward and backwards. So if there was a change from Forest to Pasture and then Pasture to Crop, only count reservoir as a Pasture to Crop change.
    - Method C:
        - Find all LULC transitions that occured within buffer of each reservoir.
        - Assign reservoir to LULC transition based on weighting system:
            Area of transition type / all ag areas in that year.
        - For example, if 10% of the buffer changes from forest to pasture and there are no other transitions that year and no other ag area, that is given a weight of 1. 
        - If the following year 5% of the buffer changes from forest to cropland, that's another 0.33 weight. 
        - Then if 5% of the buffer changes from pasture to cropland, that's 0.33 weight. 
    - Other option for method C is just to take the biggest transition in each year.


## Process:

1. bash_scripts/quick_rasterextract_allyears_allsats.sh

2. process_lulc_csvs.py

3. make_mb_transition_maps.py

4. calc_transition_summaries.py

5. annual_lc_summaries_mt.py

6. lulc_figs_w_gam.ipynb


