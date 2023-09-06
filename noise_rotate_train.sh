for ((var=0; var<3; var++))
do   
    if [ $var -eq 0 ]; then
        SEED=729
    elif [ $var -eq 1 ]; then
        SEED=492
    elif [ $var -eq 2 ]; then
        SEED=139
    elif [ $var -eq 3 ]; then
        SEED=692
    elif [ $var -eq 4 ]; then
        SEED=396
    fi
    
    for ((ROTATE=1; ROTATE<6; ROTATE++))
    do
        MODEL=ninasr_b0
        TODAY=$(date +"%Y%m%d")
        SHIFT=1
        NOISE=ROTATE
        BS=$1
        if [ $BS -eq 256 ]; then
            LR=1e-3
        elif [ $BS -eq 2048 ]; then
            LR=1e-4
        fi
        CHECK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOSIE}${ROTATE}_${var}
        WORK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}${ROTATE}_${var}
        python -m torchsr.train --arch $MODEL \
                                --scale 2 \
                                --batch-size ${BS} \
                                --lr ${LR} \
                                --epochs 300 \
                                --noise ${NOISE} \
                                --noise-value ${ROTATE} \
                                --loss l1 \
                                --dataset-train div2k_bicubic \
                                --save-checkpoint ./workdirs_230808/${MODEL}/${NOISE}/${WORK_FILE}.pt \
                                --seed ${SEED} \
                                --log ./workdirs_230808/${MODEL}/${NOISE}/${WORK_FILE}.txt
    done
done 
