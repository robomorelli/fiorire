#!/bin/bash

#/davinci-1/home/morellir/artificial_intelligence/repos/fiorire/launch_wrapper.sh num_nodes 2 num_gpus 1 num_cpus 16 model_name conv_ae1D

# Default values
NUM_NODES=-1
NUM_GPUS=-1
NUM_CPUS=-1
MODEL_NAME="conv_ae1D"

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
    model_name)
      MODEL_NAME="$value"
      shift 2
      ;;
    *)
      echo "Unknown option: $key"
      exit 1
      ;;
  esac
done

# Validate required values (use better defaults or error out)
if [[ "$NUM_NODES" -lt 1 || "$NUM_GPUS" -lt 1 || "$NUM_CPUS" -lt 1 || -z "$MODEL_NAME" ]]; then
  echo "Error: Invalid or missing values."
  echo "Usage: ./submit_job.sh num_nodes 2 num_gpus 1 num_cpus 16 model_name conv_ae1D"
  exit 1
fi

# Build PBS script
PBS_JOB="/davinci-1/home/morellir/artificial_intelligence/repos/fiorire/launch_hpo_temp.pbs"

cat > "$PBS_JOB" <<EOF
#!/bin/bash
#PBS -N fiorire
#PBS -o fiorire.log
#PBS -e ray_hpo.err
#PBS -q gpu
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=${NUM_NODES}:ngpus=${NUM_GPUS}:ncpus=${NUM_CPUS},walltime=72:00:00
#PBS -v num_nodes=${NUM_NODES},num_gpus=${NUM_GPUS},num_cpus=${NUM_CPUS},model_name=${MODEL_NAME}

bash /davinci-1/home/morellir/artificial_intelligence/repos/fiorire/launch_hpo.sh
EOF

echo "Submitting job with:"
echo "- Nodes: $NUM_NODES"
echo "- GPUs per node: $NUM_GPUS"
echo "- CPUs per node: $NUM_CPUS"
echo "- Model: $MODEL_NAME"

qsub "$PBS_JOB"