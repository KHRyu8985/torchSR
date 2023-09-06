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
    
    for ((DISTORT=1; DISTORT<6; DISTORT++))
    do
        MODEL=ninasr_b0
        TODAY=$(date +"%Y%m%d")
        SHIFT=1
        ROTATE=1
        NOISE=DISTORT
        BS=$1
        if [ $BS -eq 256 ]; then
            LR=1e-3
        elif [ $BS -eq 512 ]; then
            LR=5e-4
        elif [ $BS -eq 1024 ]; then
            LR=2e-4
        elif [ $BS -eq 2048 ]; then
            LR=1e-4
        fi
        CHECK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOSIE}${DISTORT}_${var}
        WORK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}${DISTORT}_${var}
        python -m torchsr.train --arch $MODEL \
                                --scale 2 \
                                --batch-size ${BS} \
                                --lr ${LR} \
                                --epochs 300 \
                                --noise ${NOISE} \
                                --noise-value ${DISTORT} \
                                --loss l1 \
                                --dataset-train div2k_bicubic \
                                --save-checkpoint ./workdirs_230808/${MODEL}/${NOISE}/${WORK_FILE}.pt \
                                --seed ${SEED} \
                                --log ./workdirs_230808/${MODEL}/${NOISE}/${WORK_FILE}.txt
    done
done 
