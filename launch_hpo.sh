#!/bin/bash

# Load required modules
module load openmpi
module load proxy/proxy_20

cd /davinci-1/home/morellir/artificial_intelligence/repos/fiorire

# Read from environment (passed via PBS -v)
NUM_NODES=${num_nodes}
NUM_GPUS=${num_gpus}
NUM_CPUS=${num_cpus}
# Python defaults
CONFIG_FILE=${model_name}
NUM_SAMPLES=${num_samples}
ENTITY=${entity}
WANDB_KEY=${wandb_key}

echo "python configuration:"
echo " - Nodes: $NUM_NODES"
echo " - GPUs per node: $NUM_GPUS"
echo " - CPUs per node: $NUM_CPUS"
echo " - Model: ${CONFIG_FILE:-default from Python}"
echo " - Num Samples: ${NUM_SAMPLES:-default from Python}"
echo " - ENTITY: ${ENTITY:-default from Python}"
echo " - WANDB_KEY: ${WANDB_KEY:-default from Python}"

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
  eval \$(python /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/set_gpus_env.py --n $NUM_GPUS)
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
    eval \$(python /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/set_gpus_env.py --n $NUM_GPUS)
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

  CMD=\"python $MODEL_CONFIG_PATH --address $REDIS_ADDRESS --password $REDIS_PASSWORD\"

  echo '[MASTER] Argument origin:'
  if [[ -n \"$CONFIG_FILE\" ]]; then
    CMD+=\" --config_file $CONFIG_FILE\"
    echo ' - config_file: from shell script (CLI override)'
  else
    echo ' - config_file: using default from Python'
  fi

  if [[ -n \"$NUM_SAMPLES\" ]]; then
    CMD+=\" --num_samples $NUM_SAMPLES\"
    echo ' - num_samples: from shell script (CLI override)'
  else
    echo ' - num_samples: using default from Python'
  fi

  if [[ -n \"$WANDB_KEY\" ]]; then
    CMD+=\" --wandb_key $WANDB_KEY\"
    echo ' - wandb_key: from shell script (CLI override)'
  else
    echo ' - wandb_key: using default from Python'
  fi

  if [[ -n \"$ENTITY\" ]]; then
    CMD+=\" --entity $ENTITY\"
    echo ' - entity: from shell script (CLI override)'
  else
    echo ' - entity: using default from Python'
  fi

  echo \"[MASTER] Final command:\"
  echo \$CMD
  eval \$CMD
"
