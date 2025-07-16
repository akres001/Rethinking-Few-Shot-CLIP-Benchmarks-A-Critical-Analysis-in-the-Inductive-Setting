#!/usr/bin/env bash

# custom config
DATA=/app/datasets
TRAINER=LoRA

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
    # for SEED in 1
    do
        for SHOT in 1 2 4 8 16
        # for SHOT in 1
        do
            # DIRSAVE=/app/few_shot_unlearning/few_shot_out/results_${ATTEMPT}/${ds}/${TRAINER}/${CFG}_${SHOT}shots_${FMODEL}/seed${SEED}/
            # if [ -f "$DIR" ]; then
            #     echo "Oops! The results exist at ${DIR} (so skip this job)"
            # else
            
            cd ../models_prograddassl/Dassl.pytorch
            python3 setup.py develop
            cd ../../FSL
            
            echo "current directory: $PWD"
            
            python3 CLIP-LoRA/main.py \
                    --root_path ${DATA} \
                    --shots ${SHOT} \
                    --dataset ${ds} \
                    --seed ${SEED} \
                    --attempt ${ATTEMPT} \
                    --source_model ${FPATH} \
                    --n_iters 500 \
                    --save_results 1
                            
            # fi
        done
    done
    
done

