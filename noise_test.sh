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
    
    NN=0
    if [ $NN -eq 0 ]; then
        NOISE=SHIFT
    elif [ $NN -eq 1 ]; then
        NOISE=ROTATE
    elif [ $NN -eq 2 ]; then
        NOISE=DISTORT
    fi

    LN=0
    if [ $LN -eq 0 ]; then
        LOSS=l1
    elif [ $LN -eq 1 ]; then
        LOSS=contextual
    elif [ $LN -eq 2 ]; then
        LOSS=cobi
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
        MODEL=edsr_baseline
    elif [ $MN -eq 3 ]; then
        MODEL=carn
    elif [ $MN -eq 4 ]; then
        MODEL=rcan
    fi
    echo $MODEL

    for ((TN=0; TN<1; TN++))
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
        elif [ $TN -eq 5 ]; then
            DATASET=realsr
        elif [ $TN -eq 6 ]; then
            DATASET=imagepairs
        elif [ $TN -eq 7 ]; then
            DATASET=zoom

        fi

        echo $DATASET
        # DATASET2=div2k_bicubic
        DATASET2=div2k_bicubic
        echo $DATASET2

        # for ((VALUE=0; VALUE<1; VALUE++))
        # do
        #     TODAY=$(date +"%Y%m%d")
        #     CHECK_FILE=2x_bs${BS}_300epochs_${LOSS}_${DATASET}_${DATASET2}_${NOISE}%${VALUE}_${var}
        #     WORK_FILE=test_2x_bs${BS}_300epochs_${LOSS}_${DATASET}_${DATASET2}_${NOISE}_${var}
        #     CUDA_VISIBLE_DEVICES=$1 python -m torchsr.train --arch $MODEL \
        #                                                     --scale 2 \
        #                                                     --noise ${NOISE} \
        #                                                     --noise-value ${VALUE} \
        #                                                     --metric 2 \
        #                                                     --loss ${LOSS} \
        #                                                     --dataset-traval ${DATASET2} \
        #                                                     --dataset-val ${DATASET} \
        #                                                     --dataset-train ${DATASET} \
        #                                                     --load-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${CHECK_FILE}_best.pt \
        #                                                     --seed ${SEED} \
        #                                                     --log ./workdirs_230815/${MODEL}/${NOISE}/${WORK_FILE}.txt \
        #                                                     --validation-only \
        #                                                     --workers 0 \
        #                                                     --preload-dataset

        #                                                     # --images ./data/SRBenchmarks/benchmark/Urban100/LR_bicubic/X2/img090x2.png\
        #                                                     # --hr ./data/SRBenchmarks/benchmark/Urban100/HR/img090.png \
        # done

        for ((VALUE=5; VALUE<6; VALUE++))
        do
            TODAY=$(date +"%Y%m%d")
            CHECK_FILE=2x_bs${BS}_300epochs_${LOSS}_${DATASET}_${NOISE}%${VALUE}_patch2x2_${var}
            # CHECK_FILE=2x_bs${BS}_300epochs_${LOSS}_${DATASET}_${DATASET2}_${NOISE}%${VALUE}_patch4x4_${var}
            WORK_FILE=test_2x_bs${BS}_300epochs_${LOSS}_${DATASET}_${NOISE}_patch2x2_${var}
            # WORK_FILE=test_2x_bs${BS}_300epochs_${LOSS}_${DATASET}_${DATASET2}_${NOISE}_patch4x4_${var}
            CUDA_VISIBLE_DEVICES=$1 python -m torchsr.train --arch $MODEL \
                                                            --scale 2 \
                                                            --lre True \
                                                            --batch-reweight False \
                                                            --noise ${NOISE} \
                                                            --noise-value ${VALUE} \
                                                            --loss l1 \
                                                            --metric 2 \
                                                            --dataset-traval ${DATASET2} \
                                                            --dataset-val ${DATASET} \
                                                            --dataset-train ${DATASET} \
                                                            --load-checkpoint ./workdirs_230817/${MODEL}/${NOISE}/${CHECK_FILE}_meta_best.pt \
                                                            --seed ${SEED} \
                                                            --log ./workdirs_230817/${MODEL}/${NOISE}/LRE/${WORK_FILE}.txt \
                                                            --validation-only \
                                                            --workers 0 \
                                                            --preload-dataset
                                                            # --images ./data/SRBenchmarks/benchmark/Urban100/LR_bicubic/X2/img090x2.png\
                                                            # --hr ./data/SRBenchmarks/benchmark/Urban100/HR/img090.png \
        done
    done
done 
