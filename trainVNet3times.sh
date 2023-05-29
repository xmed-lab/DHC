#!/bin/bash


while getopts 'e:c:t:l:' OPT; do
    case $OPT in
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    t) task=$OPTARG;;
		    l) lr=$OPTARG;;
    esac
done
echo $cuda

epoch=1000
echo $epoch

folder="Task_"${task}"_fully/"
echo $folder

#python code/train.py --task ${task} --exp ${folder}"vnet"${exp}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -ep ${epoch}
#python code/test.py --task ${task} --exp ${folder}"vnet"${exp}/fold1 -g ${cuda}
python code/evaluate_Ntimes.py --task ${task} --exp ${folder}"vnet"${exp} --folds 1
#python code/train.py --exp ${folder}${exp}${task}/fold2 --seed 666 -g ${cuda} --base_lr ${lr} -ep ${epoch}
#python code/test.py --exp ${folder}${exp}${task}/fold2 -g ${cuda}
#python code/evaluate_Ntimes.py --exp ${folder}${exp}${task} --folds 2
#python code/train.py --exp ${folder}${exp}${task}/fold3 --seed 2023 -g ${cuda} --base_lr ${lr} -ep ${epoch}
#python code/test.py --exp ${folder}${exp}${task}/fold3 -g ${cuda}

#python code/evaluate_Ntimes.py --exp ${folder}${exp}${task}
