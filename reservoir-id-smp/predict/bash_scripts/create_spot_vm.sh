satellite=$1
year=$2
vm_name="rp-${satellite}-${year}"
gcloud compute instances create $vm_name\
    --project=mmacedo-reservoirid \
    --zone=us-east1-b \
    --source-machine-image=respred-installed-v2 \
    --machine-type=t2d-standard-2 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --service-account=719590505057-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=respred-installed,image=projects/debian-cloud/global/images/debian-11-bullseye-v20231115,mode=rw,size=25,type=projects/mmacedo-reservoirid/zones/us-east1-b/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
    --metadata-from-file=shutdown-script=./spot_vm_shutdownscript.sh
    --metadata-from-file=startup-script=./spot_vm_startupscript.sh

