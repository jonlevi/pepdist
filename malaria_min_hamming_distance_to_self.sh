#!/bin/bash
#
#SBATCH --job-name=dh_full
#SBATCH --partition=componc_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=1G
#SBATCH --time=24:00:00
#SBATCH --array=1-1
#SBATCH --output=/data1/greenbab/users/levinej4/scratch/logs/sample%x.%j.%a.out
#SBATCH --error=/data1/greenbab/users/levinej4/scratch/logs/sample%x.%j.%a.err
python distance_to_self.py --proteome Malaria --metric hamming --aggregator min