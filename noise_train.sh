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

    for ((LN=0; LN<1; LN++))
    do
        if [ $LN -eq 0 ]; then
            LOSS=l1
        elif [ $LN -eq 1 ]; then
            LOSS=contextual
        elif [ $LN -eq 2 ]; then
            LOSS=cobi
        fi
            
        for ((VALUE=0; VALUE<1; VALUE++))
        do
            TODAY=$(date +"%Y%m%d")
            CHECK_FILE=2x_bs${BS}_300epochs_${LOSS}_zoom_realsr_${NOISE}%${VALUE}_${var}
            WORK_FILE=2x_bs${BS}_300epochs_${LOSS}_zoom_realsr_${NOISE}%${VALUE}_${var}
            CUDA_VISIBLE_DEVICES=$1 python -m torchsr.train --arch $MODEL \
                                                            --scale 2 \
                                                            --batch-size ${BS} \
                                                            --lr ${LR} \
                                                            --epochs 300 \
                                                            --noise ${NOISE} \
                                                            --noise-value ${VALUE} \
                                                            --loss ${LOSS} \
                                                            --dataset-train zoom \
                                                            --dataset-traval realsr \
                                                            --dataset-val zoom \
                                                            --save-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${WORK_FILE}.pt \
                                                            --load-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${CHECK_FILE}_e150.pt \
                                                            --seed ${SEED} \
                                                            --log ./workdirs_230809/${MODEL}/${NOISE}/${WORK_FILE}.txt \
                                                            --workers 0 \
                                                            --preload-dataset
            # CUDA_VISIBLE_DEVICES=$3 python -m torchsr.train --arch $arch \
            #                                                 --scale $scale \
            #                                                 --tune-backend \
            #                                                 --log-dir logs_train/${arch}_x${scale} \
            #                                                 --save-checkpoint checkpoint/${arch}/${arch}_x${scale}.pt \
            #                                                 --lr $learning_rate \
            #                                                 --epochs $epochs \
            #                                                 --lr-decay-steps $((epochs*2/3)) $((epochs*5/6)) \
            #                                                 --lr-decay-rate 3 \
            #                                                 --patch-size-train $(( (patch_size+1) * scale)) \
            #                                                 --shave-border $scale \
            #                                                 --replication-pad 4 \
           #                                                 --weight-norm
        done
    done
done