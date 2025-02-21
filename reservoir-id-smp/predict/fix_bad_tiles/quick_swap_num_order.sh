# Fix for bad projection originally
for f in *
    do mv $f $(echo $f | sed -r 's/^old-pred_([0-9]*)-([0-9]*).*/pred_\2-\1.tif/')
done
