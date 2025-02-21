for f in $(gcloud compute instances list | grep TERMINATED | grep rp-landsat | awk '{split($1,a,/[ ]/); print a[1]}')
do 
    gcloud compute instances start $f
done
