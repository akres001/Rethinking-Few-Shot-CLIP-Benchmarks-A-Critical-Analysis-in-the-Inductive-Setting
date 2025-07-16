#!/bin/bash

IMAGE_TEXT=1
REG=1
SPLT_GEN=0
REG_WEIGHT=1 # 1
REG_TYPE=2 # 1
MAX_EPOCHS=30
N_SAMPLES=5
NORMEMB=0
PHASES=''
IMP_STATS=1
# ARCH='RN50'
PARAMV=4
VALIDSTOP=0
RUNDS="StanfordDogs,StanfordCars,Caltech101,OxfordFlowers"

GEN_DATA=$1
ATTEMPT=$2
SEED=$3
LOSSTYPE=$4
IMP=$5
DEBUG=$6 
GSSNOISE=$7 
N_SAMPLES=$8
MINSTOPACC=$9
RUNDS=$10
SPLT_GEN=$11
PHASES=$12
ARCH=$13


DIR=/app/few_shot_unlearning/results/results${ATTEMPT}/seed_${SEED}/
if [ -e "${DIR}results_${ds}.pkl" ]; then
    echo "Oops! The results exist at '${DIR}results_${ds}.pkl' (so skip this job)"
else
    echo "Saving dir ${DIR}"
    # echo "Attempt ${ATTEMPT} DS ${ds} seed ${SEED} model ${MODEL}"
    python3 train_forget.py \
    --output_dir /app/few_shot_unlearning/results/results${ATTEMPT}/seed_${SEED}/ \
    --seed ${SEED} \
    --generated_data ${GEN_DATA} \
    --image_text ${IMAGE_TEXT} \
    --norm_embed ${NORMEMB} \
    --loss_type ${LOSSTYPE} \
    --use_importance ${IMP} \
    --use_lreg ${REG} \
    --split_train_val_generated ${SPLT_GEN} \
    --debug ${DEBUG} \
    --reg_weight ${REG_WEIGHT} \
    --reg_type ${REG_TYPE} \
    --max_epochs ${MAX_EPOCHS} \
    --n_samples ${N_SAMPLES} \
    --run_ds ${RUNDS} \
    --gaussian_noise ${GSSNOISE} \
    --min_train_acc_stop ${MINSTOPACC} \
    --phases ${PHASES} \
    --backbone_arch ${ARCH} \
    --save_importance_stats ${IMP_STATS} \
    --version_params ${PARAMV} \
    --use_valid_stop ${VALIDSTOP}
    
fi
