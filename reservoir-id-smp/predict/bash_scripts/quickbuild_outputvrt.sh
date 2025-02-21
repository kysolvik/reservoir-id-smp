file_name=$1

mkdir -p temp_vrts

gdalbuildvrt -srcnodata 255 -vrtnodata 255 temp_vrts/temp_0.vrt ./reservoir-id-smp/reservoir-id-smp/predict/out/pred_0*.tif

for i in {10..99};do
	echo $i
	gdalbuildvrt -srcnodata 255 -vrtnodata 255 temp_vrts/temp_${i}.vrt ./reservoir-id-smp/reservoir-id-smp/predict/out/pred_${i}*.tif
done

gdalbuildvrt -srcnodata 255 -vrtnodata 255 ${file_name}.vrt temp_vrts/temp_*.vrt
gdal_translate -co "COMPRESS=LZW" ${file_name}.vrt ${file_name}.tif
