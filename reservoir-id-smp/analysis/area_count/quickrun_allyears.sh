for f in ../clip/out/*aea.tif;
do 
    f_base=${f##*/}
    outfile="out/${f_base/.tif/_counts.csv}"
    python3 calc_areas_regions.py $f ./data/munis_raster_aea.tif $outfile 
done
