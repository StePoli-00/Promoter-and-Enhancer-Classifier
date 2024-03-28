#!/bin/sh
#SBATCH --job-name=<job-name>
#SBATCH --account=ai4bio2023
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
python <filename>
