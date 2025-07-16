#!/usr/bin/env bash

# custom config
DATA=/app/datasets
TRAINER=SEP
WEIGHT=8.0 # weight of the textual consistency 
WEIGHT_V=6.0 # weight of the visual consistency

CTP=end  # class token position (end or middle)
NCTX=6 # length of textual prompts
NCTX_V=4 # length of visual prompts

SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
IND=0 # indext of selected layers. 0: [1,2,3,4,5,6,7,8,9,10,11]


CFG=$1  # config file
ATTEMPT=$2
FMODEL=$3
LISTDS=$4
MODSOURCE=$5

IFS=',' read -r -a datasets <<< "$LISTDS"

if [[ "$MODSOURCE" == "scratch_unlearned" ]]; then
    MODSOURCEPATH=scratch_unlearned_cp.sh
elif [[ "$MODSOURCE" == "scratch_heldout" ]]; then
    MODSOURCEPATH=scratch_heldout_cp.sh
elif [[ "$MODSOURCE" == "original_unlearned" ]]; then
    MODSOURCEPATH=original_unlearned_cp.sh
fi


if [[ "$MODSOURCE" != "" ]]; then
    echo "loading from " $MODSOURCEPATH
    source ${MODSOURCEPATH}
else
    echo "Not loading, original model"
fi



for ds in "${datasets[@]}";
do
    if [[ "$FMODEL" == 1 ]]; then
        FPATH=${modelspath[$ds]}
        echo "Using Forget Model Path:" $FPATH
    else
        FPATH="''"
    fi
    

    for SEED in 1 2 3
    do
        for SHOT in 1 2 4 8 16
        do
            DIR=../few_shot_out/results_${ATTEMPT}/${ds}/${TRAINER}/${CFG}_${SHOT}shots_${FMODEL}/seed${SEED}/log.txt
            DIRSAVE=../few_shot_out/results_${ATTEMPT}/${ds}/${TRAINER}/${CFG}_${SHOT}shots_${FMODEL}/seed${SEED}/
            if [ -f "$DIR" ]; then
                echo "Oops! The results exist at ${DIR} (so skip this job)"
            else
            
            
            cd ../models_prograddassl/SEP.Dassl.pytorch
            python3 setup.py develop
            cd ../../FSL
            
            python3 train.py \
                    --root ${DATA} \
                    --seed ${SEED} \
                    --trainer ${TRAINER} \
                    --dataset-config-file configs/datasets/${ds}.yaml \
                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                    --output-dir ${DIRSAVE} \
                    DATASET.NUM_SHOTS ${SHOT} \
                    TRAINER.COOP.N_CTX_V ${NCTX_V} \
                    TRAINER.COOP.L_IND ${IND} \
                    TRAINER.COOP.W_V ${WEIGHT_V} \
                    TRAINER.COOP.W ${WEIGHT} \
                    MODEL.FORGET_PATH ${FPATH} \
                    TRAINER.COOP.N_CTX ${NCTX} \
                    TRAINER.COOP.CSC ${CSC} \
                    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
            fi
        done
    done
    # cd ..
    python3 parse_results.py \
    --directory ../few_shot_out/results_${ATTEMPT}/${ds}/${TRAINER}/ \
    --out_directory ../few_shot_out/results_${ATTEMPT}/${ds}/${TRAINER}_${FMODEL}_summary.pkl \
    --multi-exp
done

