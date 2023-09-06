for ((var=0; var<1; var++))
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

    NN=$2
    if [ $NN -eq 0 ]; then
        NOISE=SHIFT
    elif [ $NN -eq 1 ]; then
        NOISE=ROTATE
    elif [ $NN -eq 2 ]; then
        NOISE=DISTORT
    fi

    for ((VALUE=0; VALUE<6; VALUE++))
    do
        MODEL=ninasr_b0
        TODAY=$(date +"%Y%m%d")
        NOISE=SHIFT
        BS=$1
        if [ $BS -eq 256 ] ; then
            LR=1e-3
        elif [ $BS -eq 2048 ]; then
            LR=1e-4
        fi
        CHECK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}${VALUE}_${var}
        WORK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}${VALUE}_lre_${var}
        python -m torchsr.train --arch $MODEL \
                                --scale 2 \
                                --batch-size ${BS} \
                                --lr ${LR} \
                                --epochs 300 \
                                --lre True \
                                --batch-reweight False \
                                --noise ${NOISE} \
                                --noise-value ${VALUE} \
                                --loss l1 \
                                --dataset-train div2k_bicubic \
                                --save-checkpoint ./workdirs_230808/${MODEL}/${NOISE}/LRE/${WORK_FILE}.pt \
                                --load-checkpoint ./workdirs_230808/${MODEL}/${NOISE}/${CHECK_FILE}_e150.pt \
                                --seed ${SEED} \
                                --log ./workdirs_230808/${MODEL}/${NOISE}/LRE/${WORK_FILE}.txt
    done
done 
