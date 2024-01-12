# Fix for bad projection originally
for f in ./out/*
    do mv $f $(echo $f | sed -r 's/^.*pred_([0-9]*)-([0-9]*).*/out\/pred_\2-\1.tif/')
done
