# Fix for bad projection originally
mkdir -p out2
for f in ./out/*
    do mv $f $(echo $f | sed -r 's/^.*pred_([0-9]*)-([0-9]*).*/out2\/pred_\2-\1.tif/')
done
