# Landsat 5
for y in $(seq 1994 2011)
do
    tsp bash full_cloud_run.sh $y 5
done

# Landsat 7
for y in $(seq 2000 2023)
do
    tsp bash full_cloud_run.sh $y 7
done

# Landsat 8
for y in $(seq 2014 2023)
do
    tsp bash full_cloud_run.sh $y 8
done
