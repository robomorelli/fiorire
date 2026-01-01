#!/bin/bash

# ============================================================================
# Launch wrapper for PBS jobs
# ============================================================================
# Usage:
#   sh launch_wrapper.sh \
#       num_nodes 2 \
#       num_gpus 1 \
#       num_cpus 16 \
#       config_file conv_ae2D \
#       num_samples 100 \
#       project_name fiorire1_2D \
#       wandb 1 \
#       debug 0
# ============================================================================

# Default values
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-1}
NUM_CPUS=${NUM_CPUS:-12}
CONFIG_FILE=""
NUM_SAMPLES=""
ENTITY=""
WANDB=""
WANDB_KEY=""
PROJECT_NAME=""
DEBUG="0"  # ✅ Default: 0 (non-debug mode)

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  value="$2"
  case $key in
    num_nodes)
      NUM_NODES="$value"
      shift 2
      ;;
    num_gpus)
      NUM_GPUS="$value"
      shift 2
      ;;
    num_cpus)
      NUM_CPUS="$value"
      shift 2
      ;;
    config_file)
      CONFIG_FILE="$value"
      shift 2
      ;;
    num_samples)
      NUM_SAMPLES="$value"
      shift 2
      ;;
    entity)
      ENTITY="$value"
      shift 2
      ;;
    wandb)
      WANDB="$value"
      shift 2
      ;;
    project_name)
      PROJECT_NAME="$value"
      shift 2
      ;;
    wandb_key)
      WANDB_KEY="$value"
      shift 2
      ;;
    debug)  # ✅ New: debug mode
      DEBUG="$value"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

# ============================================================================
# Validate required arguments
# ============================================================================
if [[ -z "$CONFIG_FILE" ]]; then
  echo ""
  echo "="*80
  echo "❌ ERROR: config_file argument is required!"
  echo "="*80
  echo ""
  echo "Usage:"
  echo "  sh launch_wrapper.sh \\"
  echo "      num_nodes 2 \\"
  echo "      num_gpus 1 \\"
  echo "      num_cpus 16 \\"
  echo "      config_file conv_ae2D \\"
  echo "      num_samples 100 \\"
  echo "      project_name my_project \\"
  echo "      wandb 1 \\"
  echo "      debug 0"
  echo ""
  exit 1
fi

# ============================================================================
# Build PBS environment variables
# ============================================================================
PBS_ENV_VARS=""

if [[ -n "$NUM_NODES" ]]; then
  PBS_ENV_VARS+="num_nodes=${NUM_NODES},"
fi
if [[ -n "$NUM_GPUS" ]]; then
  PBS_ENV_VARS+="num_gpus=${NUM_GPUS},"
fi
if [[ -n "$NUM_CPUS" ]]; then
  PBS_ENV_VARS+="num_cpus=${NUM_CPUS},"
fi
if [[ -n "$CONFIG_FILE" ]]; then
  PBS_ENV_VARS+="config_file=${CONFIG_FILE},"
fi
if [[ -n "$NUM_SAMPLES" ]]; then
  PBS_ENV_VARS+="num_samples=${NUM_SAMPLES},"
fi
if [[ -n "$ENTITY" ]]; then
  PBS_ENV_VARS+="entity=${ENTITY},"
fi
if [[ -n "$WANDB_KEY" ]]; then
  PBS_ENV_VARS+="wandb_key=${WANDB_KEY},"
fi
if [[ -n "$WANDB" ]]; then
  PBS_ENV_VARS+="wandb=${WANDB},"
fi
if [[ -n "$PROJECT_NAME" ]]; then
  PBS_ENV_VARS+="project_name=${PROJECT_NAME},"
fi
if [[ -n "$DEBUG" ]]; then  # ✅ Pass debug
  PBS_ENV_VARS+="debug=${DEBUG},"
fi

# Remove trailing comma
PBS_ENV_VARS=${PBS_ENV_VARS%,}

# ============================================================================
# Create unique PBS job file (with timestamp and config name)
# ============================================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PBS_JOB="/davinci-1/home/morellir/artificial_intelligence/repos/fiorire/pbs_jobs/launch_hpo_${CONFIG_FILE}_${TIMESTAMP}.pbs"

# Create directory for PBS jobs
mkdir -p /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/pbs_jobs

# ============================================================================
# Build PBS script
# ============================================================================
cat > "$PBS_JOB" <<EOF
#!/bin/bash
#PBS -N fiorire_${CONFIG_FILE}
#PBS -o fiorire_${CONFIG_FILE}_${TIMESTAMP}.log
#PBS -e fiorire_${CONFIG_FILE}_${TIMESTAMP}.err
#PBS -q gpu
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=${NUM_NODES}:ngpus=${NUM_GPUS}:ncpus=${NUM_CPUS},walltime=72:00:00
#PBS -v $PBS_ENV_VARS

module load proxy/proxy_20
bash /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/launch_hpo.sh
EOF

# ============================================================================
# Display configuration and submit
# ============================================================================
echo ""
echo "="*80
echo "🚀 SUBMITTING PBS JOB"
echo "="*80
echo ""
echo "PBS Job file: $PBS_JOB"
echo ""
echo "Configuration:"
echo "  - Config file: ${CONFIG_FILE} ✅"
echo "  - Nodes: ${NUM_NODES}"
echo "  - GPUs per node: ${NUM_GPUS}"
echo "  - CPUs per node: ${NUM_CPUS}"
echo "  - Num Samples: ${NUM_SAMPLES:-default}"
echo "  - Project: ${PROJECT_NAME:-default}"
echo "  - Entity: ${ENTITY:-default}"
echo "  - W&B: ${WANDB:-default}"
echo "  - Debug mode: ${DEBUG} 🐛"  # ✅ Show debug status
echo ""

# Submit job
JOB_ID=$(qsub "$PBS_JOB")
echo "✅ Job submitted: $JOB_ID"
echo "📁 PBS script: $PBS_JOB"
echo ""