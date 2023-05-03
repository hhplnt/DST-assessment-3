#!/bin/bash
#SBATCH --account=math027744
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-2:00:00
#SBATCH --mem=64G
#SBATCH --job-name=gpujob
#SBATCH --error=test_stderr_gpu.txt


source ~/.bashrc # Put the node in the same state that we are in interactively
conda activate tf-env
date
cd $SLURM_SUBMIT_DIR
echo "Entered directory: `pwd`"
a=$(date +%s%N)
#sleep 1.234
python tune.py
b=$(date +%s%N)
diff=$((b-a))
date
printf "%s.%s seconds passed\n" "${diff:0: -9}" "${diff: -9:3}"