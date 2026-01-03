#!/bin/bash

# ============================================================================
# Launch HPO fine-tuning with Ray cluster
# ============================================================================
# This script is executed BY PBS on allocated nodes for FINE-TUNING
# ============================================================================

# Load required modules
module load openmpi
module load proxy/proxy_20

cd /davinci-1/home/morellir/artificial_intelligence/repos/fiorire

# ============================================================================
# Read PBS environment variables
# ============================================================================
NUM_NODES=${num_nodes}
NUM_GPUS=${num_gpus}
NUM_CPUS=${num_cpus}
TRIALS_PER_NODE=${trials_per_node:-1}
CONFIG_FILE=${config_file}
NUM_SAMPLES=${num_samples}
ENTITY=${entity}
WANDB_KEY=${wandb_key}
PROJECT_NAME=${project_name}
WANDB=${wandb}
DEBUG=${debug:-0}

# ============================================================================
# CRITICAL VALIDATION: CONFIG_FILE must not be empty
# ============================================================================
if [[ -z "$CONFIG_FILE" ]]; then
  echo ""
  echo "="*80
  echo "❌ CRITICAL ERROR: CONFIG_FILE is empty or not set!"
  echo "="*80
  echo ""
  echo "PBS variables received:"
  echo "-------------------------------------------"
  env | grep -E "^(config_file|model_name|num_|entity|wandb|project|debug|trials)" | sort
  echo "-------------------------------------------"
  echo ""
  echo "Possible causes:"
  echo "  1. launch_wrapper_ft.sh didn't pass 'config_file' argument"
  echo "  2. Variable name mismatch in PBS -v list"
  echo "  3. PBS variable not exported correctly"
  echo ""
  echo "Expected variable: config_file=<value>"
  echo "Actual value: config_file='${config_file}'"
  echo ""
  echo "Please verify launch_wrapper_ft.sh passes:"
  echo "  sh launch_wrapper_ft.sh ... config_file conv_ae2D_ft"
  echo ""
  exit 1
fi

# ============================================================================
# Optional validation: check if config YAML file exists
# ============================================================================
CONFIG_YAML_PATH="/davinci-1/home/morellir/artificial_intelligence/repos/fiorire/train_configurations/${CONFIG_FILE}.yaml"
if [[ ! -f "$CONFIG_YAML_PATH" ]]; then
  echo ""
  echo "="*80
  echo "⚠️  WARNING: Config YAML not found!"
  echo "="*80
  echo ""
  echo "Expected path: $CONFIG_YAML_PATH"
  echo "Config file: ${CONFIG_FILE}.yaml"
  echo ""
  echo "Python will try to load it, but may fail."
  echo "Continuing anyway..."
  echo ""
fi

# ============================================================================
# Display configuration
# ============================================================================
echo ""
echo "="*80
echo "🐍 PYTHON CONFIGURATION (FINE-TUNING)"
echo "="*80
echo "  - Nodes: $NUM_NODES"
echo "  - GPUs per node: $NUM_GPUS"
echo "  - CPUs per node: $NUM_CPUS"
echo "  - Trials per node: $TRIALS_PER_NODE"
echo "  - Config file: ${CONFIG_FILE} ✅"
echo "  - Num Samples: ${NUM_SAMPLES:-default}"
echo "  - Entity: ${ENTITY:-default}"
echo "  - Project: ${PROJECT_NAME:-default}"
echo "  - W&B: ${WANDB:-default}"
echo "  - Debug mode: ${DEBUG} 🐛"
echo ""

# ============================================================================
# Discover allocated nodes
# ============================================================================
NODES=($(sort -u $PBS_NODEFILE))
MASTER_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")
NUM_ACTUAL_NODES=${#NODES[@]}

echo "="*80
echo "📡 RAY CLUSTER SETUP"
echo "="*80
echo "Allocated nodes: ${NODES[*]}"
echo "Master node: $MASTER_NODE"
echo "Worker nodes: ${WORKER_NODES[*]}"
echo ""

# ============================================================================
# Ray environment setup
# ============================================================================
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
  echo "❌ ERROR: No free Redis port found on $MASTER_NODE"
  exit 1
fi

echo "✅ Redis address: $REDIS_ADDRESS"
echo ""

# ============================================================================
# Define cleanup function
# ============================================================================
cleanup_ray_cluster() {
  echo ""
  echo "="*80
  echo "🧹 CLEANUP: Stopping Ray on all nodes"
  echo "="*80

  for NODE in "${NODES[@]}"; do
    echo "[CLEANUP] Stopping Ray on $NODE"
    ssh $NODE "
      source ~/.bashrc
      conda activate fiorire
      export TMPDIR='/tmp/ray-$USER'
      CURRENT_CLUSTER=\$(cat \$TMPDIR/ray_current_cluster 2>/dev/null || true)
      if [[ \"\$CURRENT_CLUSTER\" == \"$REDIS_ADDRESS\" ]]; then
        echo \"  → Stopping Ray (owned cluster)\"
        ray stop
      else
        echo \"  → Skipped (not matching cluster)\"
      fi
    " &
  done

  wait
  echo "✅ Ray cleanup complete"
  echo ""
}

# Trap exit and signals
trap cleanup_ray_cluster EXIT SIGINT SIGTERM

# ============================================================================
# Start Ray head
# ============================================================================
echo "="*80
echo "🚀 STARTING RAY HEAD"
echo "="*80
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

# ============================================================================
# Start Ray workers
# ============================================================================
if [[ ${#WORKER_NODES[@]} -gt 0 ]]; then
  echo ""
  echo "="*80
  echo "🔗 STARTING RAY WORKERS"
  echo "="*80

  for WORKER in "${WORKER_NODES[@]}"; do
    echo "[WORKER] Starting on $WORKER"
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
fi

# ============================================================================
# Run fine-tuning
# ============================================================================
MODEL_CONFIG_PATH="main.py"

echo ""
echo "="*80
echo "🏃 RUNNING FINE-TUNING"
echo "="*80
echo "Script: $MODEL_CONFIG_PATH"
echo "Config: $CONFIG_FILE ✅ (validated)"
echo "Debug mode: $DEBUG 🐛"
echo ""

ssh $MASTER_NODE "
  source ~/.bashrc
  conda activate fiorire
  export TMPDIR='/tmp/ray-$USER'
  module load proxy/proxy_20
  cd /davinci-1/home/morellir/artificial_intelligence/repos/fiorire

  # Build command - CONFIG_FILE is ALWAYS passed (validated above)
  CMD=\"python main.py --address $REDIS_ADDRESS --password $REDIS_PASSWORD\"
  CMD+=\" --config_file $CONFIG_FILE\"
  CMD+=\" --debug_mode $DEBUG\"
  CMD+=\" --n_gpus $NUM_GPUS\"
  CMD+=\" --n_cpus $NUM_CPUS\"
  CMD+=\" --trials_per_node $TRIALS_PER_NODE\"

  # Optional arguments
  [[ -n \"$NUM_SAMPLES\" ]] && CMD+=\" --num_samples $NUM_SAMPLES\"
  [[ -n \"$WANDB_KEY\" ]] && CMD+=\" --wandb_key $WANDB_KEY\"
  [[ -n \"$ENTITY\" ]] && CMD+=\" --entity $ENTITY\"
  [[ -n \"$WANDB\" ]] && CMD+=\" --wandb $WANDB\"
  [[ -n \"$PROJECT_NAME\" ]] && CMD+=\" --project_name $PROJECT_NAME\"

  echo \"\"
  echo \"Full command:\"
  echo \"-------------------------------------------\"
  echo \"\$CMD\"
  echo \"-------------------------------------------\"
  echo \"\"

  eval \$CMD
"

# Ray will be cleaned up by the trap on exit