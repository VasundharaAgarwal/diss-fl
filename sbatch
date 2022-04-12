#!/bin/zsh
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --job-name=diss_fl

source /nfs-share/va308/diss-fl/env/bin/activate 

zs=(0 0.2 0.5 0.75 1)

for seed in {0..4}
    do
    for z in "${zs[@]}"
    do
        command="python3 main.py --noise_multiplier $z --seed $seed --results_dir /nfs-share/va308/diss-fl/results"
        eval "$command"
    done
done