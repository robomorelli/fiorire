#!/bin/bash

#PBS -N fiorire
#PBS -o fiorire.log
#PBS -e ray_hpo.err
#PBS -q gpu
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=1:ngpus=4:ncpus=48:mpiprocs=1:mem=0,walltime=72:00:00

# Load necessary modules
module load openmpi

if [ -z "$model_name" ]; then
  echo "Using default model confing file"
fi

# Get list of nodes
NODES=($(sort -u $PBS_NODEFILE))
MASTER_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")
NUM_NODES=${#NODES[@]}

echo "Allocated nodes: ${NODES[@]}"
echo "Master node: $MASTER_NODE"
echo "Worker nodes: ${WORKER_NODES[@]}"

# Get IP address of master node
MASTER_IP=$(ssh $MASTER_NODE "hostname -I | awk '{print \$1}'")
REDIS_PORT=6379
REDIS_ADDRESS="$MASTER_IP:$REDIS_PORT"
REDIS_PASSWORD="5241590000000000"

echo "Redis head address: $REDIS_ADDRESS"

# Start Ray head
echo "[MASTER] Starting Ray head on $MASTER_NODE ($MASTER_IP)"
ssh $MASTER_NODE "
source ~/.bashrc
conda activate ray
ray start --head --node-ip-address=$MASTER_IP --port=$REDIS_PORT --redis-password=$REDIS_PASSWORD
" &

sleep 10 # wait a bit to ensure the head node starts

# Start Ray workers
for WORKER in "${WORKER_NODES[@]}"; do
echo "[WORKER] Starting Ray worker on $WORKER"
ssh $WORKER "
source ~/.bashrc
conda activate ray
WORKER_IP=\$(hostname -I | awk '{print \$1}')
ray start --address=$REDIS_ADDRESS --redis-password=$REDIS_PASSWORD --node-ip-address=\$WORKER_IP
" &
done

wait # ensure all ray nodes are up before starting experiment

# Run your training script from master node
MODEL_CONFIG_PATH="/davinci-1/home/morellir/artificial_intelligence/repos/fdir/main.py"
echo "[MASTER] Running main.py from master node: $MASTER_NODE"
ssh $MASTER_NODE "
source ~/.bashrc
conda activate ray
python $MODEL_CONFIG_PATH \
--address $REDIS_ADDRESS \
--password $REDIS_PASSWORD \
--config_file $model_name
"


#qsub -v model_name=conv_ae_1d /davinci-1/home/morellir/artificial_intelligence/repos/fdir/launch_hpo.sh