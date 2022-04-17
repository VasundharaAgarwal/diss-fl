source /nfs-share/va308/diss-fl/env/bin/activate 

zs=(0 0.5 0.75 1)

for i in {0..4}
    do
    for z in "${zs[@]}"
    do
        command="python3 exp1.py --noise_multiplier $z --run_num $i"
        eval "$command"
    done
done