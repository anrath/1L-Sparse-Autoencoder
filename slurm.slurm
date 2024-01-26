#!/bin/bash

# and simply output the node's hostname to
# rq1_reproduce.output
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl5sq@virginia.edu
#
#SBATCH --job-name="reproduce Job"
#SBATCH --error="reproduce.err"
#SBATCH --output="reproduce.output"
# --- specify the partition (queue) name;;; gpu  --partition="serval01"
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:2
# 
#source /etc/profile.d/modules.sh
#module load python3
#source sparse/bin/activate
cd /u/kl5sq/nlp-project/1L-Sparse-Autoencoder
python3 train.py
