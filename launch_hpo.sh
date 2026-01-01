#!/bin/bash

# Load required modules
module load openmpi
module load proxy/proxy_20

cd /davinci-1/home/morellir/artificial_intelligence/repos/fiorire

# Read PBS environment
NUM_NODES=${num_nodes}
NUM_GPUS=${num_gpus}
NUM_CPUS=${num_cpus}
CONFIG_FILE=${model_name}
NUM_SAMPLES=${num_samples}
ENTITY=${entity}
WANDB_KEY=${wandb_key}
PROJECT_NAME=${project_name}
WANDB=${wandb}

echo "python configuration:"
echo " - Nodes: $NUM_NODES"
echo " - GPUs per node: $NUM_GPUS"
echo " - CPUs per node: $NUM_CPUS"
echo " - Model: ${CONFIG_FILE:-default}"
echo " - Num Samples: ${NUM_SAMPLES:-default}"
echo " - ENTITY: ${ENTITY:-default}"
echo " - WANDB_KEY: ${WANDB_KEY:-default}"
echo " - PROJECT_NAME: ${PROJECT_NAME:-default}"
echo " - WANDB: ${WANDB:-default}"

# Discover nodes
NODES=($(sort -u $PBS_NODEFILE))
MASTER_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")
NUM_ACTUAL_NODES=${#NODES[@]}
echo "Allocated nodes: ${NODES[*]}"
echo "Master node: $MASTER_NODE"
echo "Worker nodes: ${WORKER_NODES[*]}"

# Ray environment setup
REDIS_PASSWORD="5241590000000000"
TMPDIR="/tmp/ray-$USER"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

# Auto-select Redis port on master node
MASTER_IP=$(ssh $MASTER_NODE "hostname -I | awk '{print \$1}'")
for port in {6379..6399}; do
  ssh $MASTER_NODE "lsof -i :$port" &> /dev/null
  if [[ $? -ne 0 ]]; then
    REDIS_PORT=$port
    REDIS_ADDRESS="$MASTER_IP:$REDIS_PORT"
    break
  fi
done
if [[ -z "$REDIS_PORT" ]]; then
  echo "[ERROR] No free Redis port found on $MASTER_NODE"
  exit 1
fi

# Define cleanup function
cleanup_ray_cluster() {
  echo "[CLEANUP] Stopping Ray on all nodes..."

  for NODE in "${NODES[@]}"; do
    echo "[CLEANUP] Checking and stopping Ray on $NODE"
    ssh $NODE "
      source ~/.bashrc
      conda activate fiorire
      export TMPDIR='/tmp/ray-$USER'
      CURRENT_CLUSTER=\$(cat \$TMPDIR/ray_current_cluster 2>/dev/null || true)
      if [[ \"\$CURRENT_CLUSTER\" == \"$REDIS_ADDRESS\" ]]; then
        echo \"[CLEANUP] Stopping Ray on $NODE (owned cluster)\"
        ray stop
      else
        echo \"[CLEANUP] Skipped Ray stop on $NODE (not matching cluster)\"
      fi
    " &
  done

  wait
  echo "[CLEANUP] Ray cleanup complete."
}

# Trap exit and signals
trap cleanup_ray_cluster EXIT SIGINT SIGTERM

# Start Ray head
echo "[MASTER] Starting Ray head on $MASTER_NODE ($MASTER_IP)"
ssh $MASTER_NODE "
  source ~/.bashrc
  conda activate fiorire
  export TMPDIR='/tmp/ray-$USER'
  module load proxy/proxy_20
  mkdir -p \$TMPDIR
  chmod 700 \$TMPDIR
  ray stop
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
    export TMPDIR='/tmp/ray-$USER'
    module load proxy/proxy_20
    mkdir -p \$TMPDIR
    chmod 700 \$TMPDIR
    ray stop
    WORKER_IP=\$(hostname -I | awk '{print \$1}')
    eval \$(python /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/set_gpus_env.py --n $NUM_GPUS)
    ray start --address=$REDIS_ADDRESS --redis-password=$REDIS_PASSWORD --node-ip-address=\$WORKER_IP
  " &
done

wait

# Run training
MODEL_CONFIG_PATH="main.py"
echo "[MASTER] Running main.py on $MASTER_NODE"
ssh $MASTER_NODE "
  source ~/.bashrc
  conda activate fiorire
  export TMPDIR='/tmp/ray-$USER'
  module load proxy/proxy_20
  cd /davinci-1/home/morellir/artificial_intelligence/repos/fiorire

  CMD=\"python $MODEL_CONFIG_PATH --address $REDIS_ADDRESS --password $REDIS_PASSWORD\"

  [[ -n \"$CONFIG_FILE\" ]] && CMD+=\" --config_file $CONFIG_FILE\"
  [[ -n \"$NUM_SAMPLES\" ]] && CMD+=\" --num_samples $NUM_SAMPLES\"
  [[ -n \"$WANDB_KEY\" ]] && CMD+=\" --wandb_key $WANDB_KEY\"
  [[ -n \"$ENTITY\" ]] && CMD+=\" --entity $ENTITY\"
  [[ -n \"$WANDB\" ]] && CMD+=\" --wandb $WANDB\"
  [[ -n \"$PROJECT_NAME\" ]] && CMD+=\" --project_name $PROJECT_NAME\"

  echo \"[MASTER] Running: \$CMD\"
  eval \$CMD
"

# Ray will be cleaned up by the trap on exit
