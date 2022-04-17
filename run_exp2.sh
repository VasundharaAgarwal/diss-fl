source /nfs-share/va308/diss-fl/env/bin/activate 


for i in {0..4}
    do
        command1="python3 exp2.py --strategy fedavg --run_num $i"
        eval "$command1"
        command2="python3 exp2.py --strategy fedavgm --run_num $i"
        eval "$command2"

    done
done