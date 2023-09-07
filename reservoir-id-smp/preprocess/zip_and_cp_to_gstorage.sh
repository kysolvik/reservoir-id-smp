# NOTE: Will need to setup up on google cloud storage bucket and 
# replace "res-id" in this code with that bucket
# https://cloud.google.com/storage/docs/creating-buckets
zip -r ../data/reservoirs_10band.zip ../data/reservoirs_10band/
gsutil cp ../data/reservoirs_10band.zip gs://res-id/cnn/training/example_prepped/
gsutil cp ../data/mean_std_v1.npy gs://res-id/cnn/training/example_prepped/
gsutil acl -r ch -u AllUsers:R gs://res-id/cnn/training/example_prepped/
