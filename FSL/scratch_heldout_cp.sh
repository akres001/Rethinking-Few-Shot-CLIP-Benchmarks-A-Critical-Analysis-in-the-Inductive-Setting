#!/usr/bin/env bash

declare -A modelspath

modelspath["dogs_imagenet"]="../models_checkpoints/scratch_heldout/imnet_nodogs.pt"
modelspath["birds_imagenet"]="../models_checkpoints/scratch_heldout/imnet_nobirds.pt"
modelspath["vehicles_imagenet"]="../models_checkpoints/scratch_heldout/imnet_novehicles.pt"

