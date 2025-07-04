import subprocess
import sys
import os
import argparse


def get_free_gpus(max_utilization=10, max_used_memory=30):
    try:
        output = subprocess.check_output(["nvidia-smi",
                                          "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                          "--format=csv,noheader,nounits"])
        gpu_info = output.decode("utf-8").strip().split("\n")
        indices = []
        for line in gpu_info:
            index, memory_used, memory_total, utilization = map(int, line.split(","))
            if (utilization < max_utilization) & (memory_used<max_used_memory*memory_total/100):
                indices.append(index)
        return indices
    except subprocess.CalledProcessError:
        # Handle error when nvidia-smi command fails
        return None


def initialize_devices(n=1, max_utilization=30, max_used_memory=30):
    # If on 'linux' get the free GPU
    if sys.platform == 'linux':
        free_gpus = get_free_gpus(max_utilization, max_used_memory)
        if free_gpus:
            #print('Free GPUs:', free_gpus)
            if len(free_gpus) < n:
                raise ValueError(
                    f'Warning: Not enough free GPUs available (requested {n}, available {len(free_gpus)})')

            gpu_visible = ''
            for i in range(n):
                gpu_visible += str(free_gpus[i]) + ','
            gpu_visible = gpu_visible[:-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible
            print(f"export CUDA_VISIBLE_DEVICES={gpu_visible}")

        else:
            raise ValueError('Impossible to set free gpus')

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=2, type=int, help="number of gpus")
    args = parser.parse_args()

    initialize_devices(args.n)
