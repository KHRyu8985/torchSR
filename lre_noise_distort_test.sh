for ((var=2; var<3; var++))
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

    for ((DISTORT=5; DISTORT<6; DISTORT++))
    do
        MODEL=ninasr_b0
        TODAY=$(date +"%Y%m%d")
        SHIFT=1
        ROTATE=1
        NOISE=DISTORT
        BS=256
        LR=1e-3
#        IMAGE=DIV2K_vallid_LR_bicubic\/X2\/0801x2\.png
        CHECK_FILE=2x_bs${BS}_300epochs_l1_div2k_bicubic_${NOISE}Albu${DISTORT}_lre_higher_${var}
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
                                --distort ${DISTORT} \
                                --loss l1 \
                                --dataset-train div2k_bicubic \
                                --load-checkpoint ./workdirs/${MODEL}/${NOISE}/${CHECK_FILE}_meta_e300.pt \
                                --seed ${SEED} \
                                --log ./workdirs/${MODEL}/${NOISE}/LRE/${CHECK_FILE}_${TODAY}.txt \
                                --images ./data/DIV2K/DIV2K_valid_LR_bicubic/X2/0801x2.png \
                                --destination ./results \
                                --hr ./data/DIV2K/DIV2K_valid_HR/0801.png \
#                                --batch-reweight False \
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
