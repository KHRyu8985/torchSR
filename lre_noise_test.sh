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
    
    NN=$1
    if [ $NN -eq 0 ]; then
        NOISE=SHIFT
    elif [ $NN -eq 1 ]; then
        NOISE=ROTATE
    elif [ $NN -eq 2 ]; then
        NOISE=DISTORT
    fi

    BS=128
    if [ $BS -eq 256 ] ; then
        LR=1e-3
    elif [ $BS -eq 128 ] ; then
        LR=1e-3
    elif [ $BS -eq 64 ] ; then
        LR=4e-3
    elif [ $BS -eq 32 ] ; then
        LR=1e-4
    elif [ $BS -eq 16 ] ; then
        LR=1e-4
    elif [ $BS -eq 2048 ]; then
        LR=1e-4
    fi

    MN=0
    if [ $MN -eq 0 ]; then
        MODEL=ninasr_b0
    elif [ $MN -eq 1 ]; then
        MODEL=vdsr
    elif [ $MN -eq 2 ]; then
        MODEL=edsr
    elif [ $MN -eq 3 ]; then
        MODEL=carn
    elif [ $MN -eq 4 ]; then
        MODEL=rcan
    fi

    for ((TN=1; TN<5; TN++))
    do
        if [ $TN -eq 0 ]; then
            DATASET=div2k_bicubic
        elif [ $TN -eq 1 ]; then
            DATASET=set5
        elif [ $TN -eq 2 ]; then
            DATASET=set14
        elif [ $TN -eq 3 ]; then
            DATASET=b100
        elif [ $TN -eq 4 ]; then
            DATASET=urban100
        fi

        echo $DATASET
        for ((VALUE=5; VALUE<6; VALUE++))
        do
            TODAY=$(date +"%Y%m%d")
            CHECK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}%${VALUE}_${var}
            WORK_FILE=test_2x_bs${BS}_300epochs_l1_${DATASET}_${NOISE}%${VALUE}_${var}
            CUDA_VISIBLE_DEVICES=$2 python -m torchsr.train --arch $MODEL \
                                                            --scale 2 \
                                                            --lre True \
                                                            --batch-reweight False \
                                                            --noise ${NOISE} \
                                                            --noise-value ${VALUE} \
                                                            --loss l1 \
                                                            --dataset-val ${DATASET} \
                                                            --load-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${CHECK_FILE}_meta_best.pt \
                                                            --seed ${SEED} \
                                                            --log ./workdirs_230809/${MODEL}/${NOISE}/LRE/${WORK_FILE}.txt \
                                                            --validation-only 
                                                            # --images ./data/SRBenchmarks/benchmark/Urban100/LR_bicubic/X2/img090x2.png\
                                                            # --hr ./data/SRBenchmarks/benchmark/Urban100/HR/img090.png \
        done
    done
done 
