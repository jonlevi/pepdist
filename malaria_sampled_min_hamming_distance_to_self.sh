#!/bin/bash
#
#SBATCH --job-name=dh_sample
#SBATCH --partition=componc_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --time=96:00:00
#SBATCH --array=1-94
#SBATCH --output=/data1/greenbab/users/levinej4/scratch/logs/sample%x.%j.%a.out
#SBATCH --error=/data1/greenbab/users/levinej4/scratch/logs/sample%x.%j.%a.err
j=$SLURM_ARRAY_TASK_ID
n=$(cat n_values.txt | awk -v ln=$j "NR==ln")
python distance_to_self.py --proteome Malaria --N $n --iter 1000 --metric hamming --aggregator min