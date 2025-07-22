#!/bin/bash

# Load required modules
module load openmpi
module load proxy/proxy_20

cd /davinci-1/home/morellir/artificial_intelligence/repos/fiorire

# Read from environment (passed via PBS -v)
NUM_NODES=${num_nodes:-1}
NUM_GPUS=${num_gpus:-1}
NUM_CPUS=${num_cpus:-12}
MODEL_NAME=${model_name:-conv_ae1D}
NUM_SAMPLES=${num_samples:-100}

echo "Job configuration:"
echo " - Nodes: $NUM_NODES"
echo " - GPUs per node: $NUM_GPUS"
echo " - CPUs per node: $NUM_CPUS"
echo " - Model: $MODEL_NAME"
echo " - Num Samples: $NUM_SAMPLES"

# Discover node list
NODES=($(sort -u $PBS_NODEFILE))
MASTER_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")
NUM_ACTUAL_NODES=${#NODES[@]}

echo "Allocated nodes: ${NODES[*]}"
echo "Master node: $MASTER_NODE"
echo "Worker nodes: ${WORKER_NODES[*]}"

# Redis for Ray
MASTER_IP=$(ssh $MASTER_NODE "hostname -I | awk '{print \$1}'")
REDIS_PORT=6379
REDIS_ADDRESS="$MASTER_IP:$REDIS_PORT"
REDIS_PASSWORD="5241590000000000"

# Start Ray head
echo "[MASTER] Starting Ray head on $MASTER_NODE ($MASTER_IP)"
ssh $MASTER_NODE "
  source ~/.bashrc
  conda activate fiorire
  module load proxy/proxy_20
  echo $NUM_GPUS
  echo $NUM_GPUS
  echo $NUM_GPUS
  echo $NUM_GPUS
  eval $(python /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/set_gpus_env.py --n $NUM_GPUS)
  ray start --head --node-ip-address=$MASTER_IP --port=$REDIS_PORT --redis-password=$REDIS_PASSWORD
" &

sleep 10

# Start Ray workers
for WORKER in "${WORKER_NODES[@]}"; do
  echo "[WORKER] Starting Ray worker on $WORKER"
  ssh $WORKER "
    source ~/.bashrc
    conda activate fiorire
    module load proxy/proxy_20
    WORKER_IP=\$(hostname -I | awk '{print \$1}')
    echo $NUM_GPUS
    echo $NUM_GPUS
    echo $NUM_GPUS
    echo $NUM_GPUS
    eval $(python /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/set_gpus_env.py --n $NUM_GPUS)
    ray start --address=$REDIS_ADDRESS --redis-password=$REDIS_PASSWORD --node-ip-address=\$WORKER_IP
  " &
done

wait

# Run Ray Tune training
MODEL_CONFIG_PATH="main.py"
echo "[MASTER] Running main.py on $MASTER_NODE"
ssh $MASTER_NODE "
  source ~/.bashrc
  conda activate fiorire
  module load proxy/proxy_20
  cd /davinci-1/home/morellir/artificial_intelligence/repos/fiorire
  python $MODEL_CONFIG_PATH \
    --address $REDIS_ADDRESS \
    --password $REDIS_PASSWORD \
    --config_file $MODEL_NAME \
    --num_samples $NUM_SAMPLES
"