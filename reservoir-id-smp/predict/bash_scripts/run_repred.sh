for satellite in ls5 ls8
do
	sat_num=${satellite:2:1} 
	tsp bash bash_scripts/prep_model_run.sh landsat${sat_num}
	for f in ~/bad_tiles/${satellite}*clean.csv
	do 
		echo $f
		bn=$(basename $f)
		y=${bn:4:4}
		out_dir=out/${satellite}_${y}_repred
		mkdir $out_dir
	    	tsp python -u repredict_smp_landsat.py \
			$f $sat_num $y ./model.ckpt ./mean_std.npy ./bands_minmax.npy \
			$out_dir --quantized --calc_nds
	done
done
