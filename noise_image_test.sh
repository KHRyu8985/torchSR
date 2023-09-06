# searchdir='./data/SRBenchmarks/benchmark/Set14/HR'
# for file in $searchdir/*
for ((var=0; var<1; var++))
do
    SEED=139
    # if [ $var -eq 0 ]; then
    #     SEED=729
    # elif [ $var -eq 1 ]; then
    #     SEED=492
    # elif [ $var -eq 2 ]; then
    #     SEED=139
    # elif [ $var -eq 3 ]; then
    #     SEED=692
    # elif [ $var -eq 4 ]; then
    #     SEED=396
    # fi
    # filename=${file##*/}
    # filen=${filename%.*}
    # if [ ${filename#*.}  = 'png' ] ; then
        
    #     echo $filen

    # else
    #     echo $filname
    #     continue
    # fi

    NN=0
    if [ $NN -eq 0 ]; then
        NOISE=SHIFT
    elif [ $NN -eq 1 ]; then
        NOISE=ROTATE
    elif [ $NN -eq 2 ]; then
        NOISE=DISTORT
    fi

    MODEL=ninasr_b0
    DATASET=div2k_bicubic

    for ((VALUE=5; VALUE<6; VALUE++))
    do
        TODAY=$(date +"%Y%m%d")
        CHECK_FILE=2x_bs128_300epochs_l1_div2k_bicubic_${NOISE}%5_${var}
        WORK_FILE=test_img_2x_bs128_300epochs_l1_${DATASET}_${NOISE}%5_${var}
        # IMG_FOLDER_LR=DIV2K/DIV2K_valid_LR_bicubic/X2
        # IMG_FOLDER_HR=DIV2K/DIV2K_valid_HR
        # IMG_FOLDER_LR=SRBenchmarks/benchmark/Set5/LR_bicubic/X2
        # IMG_FOLDER_HR=SRBenchmarks/benchmark/Set5/HR
        # IMG_FOLDER_LR=SRBenchmarks/benchmark/Set14/LR_bicubic/X2
        # IMG_FOLDER_HR=SRBenchmarks/benchmark/Set14/HR    
        # IMG_FOLDER_LR=SRBenchmarks/benchmark/Urban100/LR_bicubic/X2
        # IMG_FOLDER_HR=SRBenchmarks/benchmark/Urban100/HR
        # IMG_FOLDER_LR=SRBenchmarks/benchmark/B100/LR_bicubic/X2
        # IMG_FOLDER_HR=SRBenchmarks/benchmark/B100/HR
        IMG_FOLDER_LR=Zoom/val/00366/aligned
        IMG_FOLDER_HR=Zoom/val/00366/aligned

        # IMG_FILE=$filen  #0803, 0826, 0863, 0894 img090 baboon
        IMG_FILE=00002
        CUDA_VISIBLE_DEVICES=$1 python -m torchsr.train --arch $MODEL \
                                                        --scale 2 \
                                                        --noise ${NOISE} \
                                                        --noise-value ${VALUE} \
                                                        --loss l1 \
                                                        --seed ${SEED} \
                                                        --lre True \
                                                        --metric 0 \
                                                        --imagename $IMG_FILE \
                                                        --destination ./results_230809/CAM/ \
                                                        --images ./data/${IMG_FOLDER_LR}/${IMG_FILE}_LR.JPG \
                                                        --hr ./data/${IMG_FOLDER_HR}/${IMG_FILE}.JPG \
                                                        --log ./workdirs_230815/${MODEL}/${NOISE}/${WORK_FILE}.txt \
                                                        --load-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${CHECK_FILE}_e300.pt \
                                                        # --validation-only

    done
done 
