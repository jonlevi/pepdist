#!/bin/bash
#
#SBATCH --job-name=distances
#SBATCH --partition=cpushort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/data1/greenbab/users/levinej4/scratch/logs/NNDist%x.%j.%a.out
#SBATCH --error=/data1/greenbab/users/levinej4/scratch/logs/NNDist%x.%j.%a.err
python virus_distance_to_self_example.py