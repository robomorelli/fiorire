#!/bin/bash

#sh /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/launch_wrapper.sh num_nodes 2 num_gpus 1 num_cpus 16 config_file conv_ae1D

# Default values
# Set defaults if empty for PBS resource specification
NUM_NODES=${NUM_NODES:-1}
NUM_GPUS=${NUM_GPUS:-1}
NUM_CPUS=${NUM_CPUS:-12}
CONFIG_FILE=""
NUM_SAMPLES=""
ENTITY=""
WANDB=""
WANDB_KEY=""
PROJECT_NAME=""
#WANDB_KEY=-"56b6f7f0b13c4d89207e51c28ceb90c24201eab5"

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
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

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

# Remove trailing comma
PBS_ENV_VARS=${PBS_ENV_VARS%,}

# Build PBS script
PBS_JOB="/davinci-1/home/morellir/artificial_intelligence/repos/fiorire/launch_hpo_temp.pbs"

cat > "$PBS_JOB" <<EOF
#!/bin/bash
#PBS -N fiorire
#PBS -o fiorire.log
#PBS -e fiorire.err
#PBS -q gpu
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=${NUM_NODES}:ngpus=${NUM_GPUS}:ncpus=${NUM_CPUS},walltime=72:00:00
#PBS -v $PBS_ENV_VARS

##PBS -v num_nodes=${NUM_NODES},num_gpus=${NUM_GPUS},num_cpus=${NUM_CPUS},config_file=${config_file},num_samples=${NUM_SAMPLES},wandb_key=${WANDB_KEY},entity=${ENTITY}

module load proxy/proxy_20
bash /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/launch_hpo.sh
EOF

echo "Submitting job with:"
echo "- Nodes: ${NUM_NODES:-default (1)}"
echo "- GPUs per node: ${NUM_GPUS:-default (1)}"
echo "- CPUs per node: ${NUM_CPUS:-default (12)}"
echo "- Model: ${CONFIG_FILE:-default value from Python}"
echo "- Num Samples: ${NUM_SAMPLES:-default value from Python}"
echo "- WANDB KEY: ${WANDB_KEY:-default value from Python}"
echo "- ENTITY: ${ENTITY:-default value from Python}"
echo "- WANDB: ${WANDB:-default value from Python}"
echo "- PROJECT_NAME: ${PROJECT_NAME:-default value from Python}"


qsub "$PBS_JOB"