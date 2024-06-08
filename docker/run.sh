set -e
NAME=segue_bash_`shuf -i 2000-65000 -n 1`
CT=sheepmao/qoe-optimization
HOSTNAME=QoE-aware_Video_Chunking_Optimization

echo "Starting container name=$NAME with image $CT"
echo "Hostname --> $HOSTNAME"
echo "PWD --> $(pwd)"

sudo docker run \
    --gpus all \
    --hostname $HOSTNAME\
    --mount type=bind,source=$(pwd)/..,target=/QoE-aware_Video_Chunking_Optimization \
    --name $NAME --rm -i -t $CT bash 
