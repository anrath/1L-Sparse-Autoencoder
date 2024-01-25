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
# --- specify the partition (queue) name
#SBATCH --partition="gpu"
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=16G
#source /etc/profile.d/modules.sh
#module load python3
#source sparse/bin/activate
cd /u/kl5sq/nlp-project/1L-Sparse-Autoencoder
python3 train.py
