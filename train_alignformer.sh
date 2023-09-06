BS=128
LR=1e-3
NOISE=REAL
NOISE_VALUE=0
CHECK_FILE=1x_bs${BS}_300epochs_l1_alignformer_${NOISE}_0
WORK_FILE=1x_bs${BS}_300epochs_l1_alignformer_${NOISE}_0
MODEL=ninasr_b0
SEED=729


CUDA_VISIBLE_DEVICES=$1 python -m torchsr.train --arch ${MODEL} \
                                                --scale 1 \
                                                --seed ${SEED} \
                                                --epochs 300 \
                                                --loss l1 \
                                                --batch-size ${BS} \
                                                --lr ${LR} \
                                                --dataset-train alignformer \
                                                --dataset-traval alignformer \
                                                --dataset-val alignformer \
                                                --patch-size-train 512 \
                                                --patch-size-traval 96 \
                                                --save-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${WORK_FILE}.pt \
                                                --load-checkpoint ./workdirs_230809/${MODEL}/${NOISE}/${CHECK_FILE}_e150.pt \
                                                --log ./workdirs_230809/${MODEL}/${NOISE}/LRE/${WORK_FILE}.txt \
                                                --lre True \
                                                --batch-reweight False \
                                                # --noise ${NOISE} \
                                                # --noise-value ${NOISE_VALUE} \                                                

