#!/usr/bin/env bash

declare -A modelspath

modelspath["stanford_dogs"]="../models_checkpoints/original_unlearned/forget_StanfordDogs_retain_Caltech101|OxfordFlowers|CUB_attempt_1.pth"
modelspath["stanford_cars"]="../models_checkpoints/original_unlearned/forget_StanfordCars_retain_Caltech101|CUB|FGVCAircraft_attempt_1.pth"
modelspath["caltech101"]="../models_checkpoints/original_unlearned/forget_Caltech101_retain_StanfordDogs|OxfordFlowers|UCF101|FGVCAircraft_attempt_1.pth"
modelspath["fgvc_aircraft"]="../models_checkpoints/original_unlearned/forget_FGVCAircraft_retain_StanfordCars|Caltech101|OxfordFlowers_attempt_1.pth"
modelspath["ucf101"]="../models_checkpoints/original_unlearned/forget_UCF101_retain_StanfordDogs|Caltech101|CUB_attempt_1.pth"
modelspath["oxford_flowers"]="../models_checkpoints/original_unlearned/forget_OxfordFlowers_retain_StanfordDogs|Caltech101|CUB_attempt_1.pth"
modelspath["cub"]="../models_checkpoints/original_unlearned/forget_CUB_retain_Caltech101|OxfordFlowers|UCF101_attempt_1.pth"
