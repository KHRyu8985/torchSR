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

    NOISE=SHIFT
    for ((SHIFT=1; SHIFT<6; SHIFT++))
    do
        MODEL=ninasr_b0
        TODAY=$(date +"%Y%m%d")
        ROTATE=1
        DISTORT=1
        BS=256
        LR=1e-3
#        IMAGE=DIV2K_vallid_LR_bicubic\/X2\/0801x2\.png
        CHECK_FILE=2x_bs256_300epochs_l1_div2k_bicubic_SHIFT5_batch_lre_higher_20230723_2_e150
        # CHECK_FILE=2x_bs256_300epochs_l1_div2k_bicubic_SHIFT${SHIFT}%_0
# CHECK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}Albu${SHIFT}_lre_higher_${var}
        python -m torchsr.train --arch $MODEL \
                                --scale 2 \
                                --batch-size ${BS} \
                                --lr ${LR} \
                                --epochs 300 \
                                --lre True \
                                --index-patch True \
                                --batch-reweight False \
                                --noise ${NOISE} \
                                --shift ${SHIFT} \
                                --rotate ${ROTATE} \
                                --distort ${SHIFT} \
                                --loss l1 \
                                --dataset-train div2k_bicubic \
                                --load-checkpoint ./workdirs/${MODEL}/${NOISE}/LRE/${CHECK_FILE}.pt \
                                --seed ${SEED} \
                                --log ./workdirs/${MODEL}/${NOISE}/LRE/${CHECK_FILE}_${TODAY}.txt \
                                --images ./data/SRBenchmarks/benchmark/Urban100/LR_bicubic/X2/img090x2.png\
                                --destination ./results \
                                --hr ./data/SRBenchmarks/benchmark/Urban100/HR/img090.png \

                                # --load-checkpoint ./workdirs/${MODEL}/${NOISE}/LRE/${CHECK_FILE}.pt \

#                                --batch-reweight False \
# ./data/SRBenchmarks/benchmark/Set5/LR_bicubic/X2/butterflyx2.png
# Urban100/LR_bicubic/X2/img090x2.png
# benchmark/Set14/LR_bicubic/X2/zebrax2.png
#                        --load-checkpoint ./workdirs/${MODEL}/${CHECK_FILE}_best.pt \
#                        --batch-size 8 \
#                        --load-checkpoint ./workdirs/ninasr_b1/${CHECK_FILE}_best.pt \
#                        --validation-only \
# python -m torchsr.train --arch ninasr_b1 \
#                         --scale 2 \
#                         --epochs 300 \
#                         --loss l1 \
#                         --dataset-train div2k_bicubic \
#                         --log ./workdirs/ninasr_b1/test.txt
    done
done 
