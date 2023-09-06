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
    elif [ $BS -eq 8 ] ; then
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

    for ((VALUE=5; VALUE<6; VALUE++))
    do
        TODAY=$(date +"%Y%m%d")
        CHECK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}%${VALUE}_${var}
        WORK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}%${VALUE}_mwn3_${var} #793
        # CUDA_VISIBLE_DEVICES=$1 python -m torchsr.train --arch $MODEL \
        #                                                 --scale 2 \
        #                                                 --batch-size ${BS} \
        #                                                 --lr ${LR} \
        #                                                 --epochs 60 \
        #                                                 --noise ${NOISE} \
        #                                                 --noise-value ${VALUE} \
        #                                                 --loss l1 \
        #                                                 --dataset-train zoom \
        #                                                 --dataset-traval zoom \
        #                                                 --dataset-val zoom \
        #                                                 --save-checkpoint ./workdirs_230829/${MODEL}/${NOISE}/${WORK_FILE}.pt \
        #                                                 --load-checkpoint ./workdirs_230829/${MODEL}/${NOISE}/${CHECK_FILE}_e30.pt \
        #                                                 --seed ${SEED} \
        #                                                 --log ./workdirs_230829/${MODEL}/${NOISE}/${WORK_FILE}.txt \

        CUDA_VISIBLE_DEVICES=$1 python -m torchsr.train --arch $MODEL \
                                                        --scale 2 \
                                                        --batch-size ${BS} \
                                                        --lr ${LR} \
                                                        --epochs 300 \
                                                        --lre True \
                                                        --reweight 2 \
                                                        --noise ${NOISE} \
                                                        --noise-value ${VALUE} \
                                                        --loss l1 \
                                                        --dataset-train div2k_bicubic \
                                                        --dataset-traval div2k_bicubic \
                                                        --dataset-val div2k_bicubic \
                                                        --save-checkpoint ./workdirs_230906/${MODEL}/${NOISE}/${WORK_FILE}.pt \
                                                        --load-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${CHECK_FILE}_e150.pt \
                                                        --seed ${SEED} \
                                                        --log ./workdirs_230906/${MODEL}/${NOISE}/MWN/${WORK_FILE}.txt \
                                                        # --workers 0 \
                                                        # --preload-dataset

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
