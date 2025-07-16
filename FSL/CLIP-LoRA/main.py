import torch
import torchvision.transforms as transforms
import clip


import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.cub
import datasets.imagenet
import datasets.stanford_dogs
import datasets.dogs_imagenet
import datasets.birds_imagenet
import datasets.vehicles_imagenet
import datasets.plantdoc

from dassl.data.datasets.build import DATASET_REGISTRY, build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora

from dassl.config import get_cfg_default
from yacs.config import CfgNode as CN

torch.set_num_threads(10)


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of an aircraft {}.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of a {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "PLANTDOC" : "a photo of a {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "StanfordDogs" : "a photo of a {}.",
    "DogsImnet" : "a photo of a {}.",
    "BirdsImnet" : "a photo of a {}.",
    "CUB" : "a photo of a {}.",
    "VehiclesImnet" : "a photo of a {}.",


}

dataset_name_mapping = {
    "Caltech101": "caltech101",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc_aircraft",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "PLANTDOC" : "plantdoc",
    "UCF101": "ucf101",
    "CUB": "cub",
    "StanfordDogs" : "stanford_dogs",
    
    "DogsImnet" : "dogs_imagenet",
    "BirdsImnet" : "birds_imagenet",
    "VehiclesImnet" : "vehicles_imagenet",

}

name_dataset_mapping = {v : k for k, v in dataset_name_mapping.items()}

def main():

    # Load config file
    args = get_arguments()
    
    print("Arguments : ", args)
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    if args.source_model != "''":
        print("********** LOADING FROM FORGET PATH **************", args.source_model)
        print(clip_model.load_state_dict(torch.load(args.source_model)['model_dict']))
    
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
        
    args.config_file = "configs/trainers/TaskRes/adam_lr2e-4_B256_ep200_ViT16.yaml"
    
    cfg = get_cfg_default()


    cfg.merge_from_file(args.config_file)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = "/app/datasets/"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASET.NUM_SHOTS = args.shots

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 4
    cfg.DATALOADER.TEST.BATCH_SIZE = 10

    
    cfg.DATASET.NAME = name_dataset_mapping[args.dataset]
    
    dataset = build_dataset(cfg)
    
    dataset.template = [CUSTOM_TEMPLATES[name_dataset_mapping[args.dataset]]]
    
    tfm_train = build_transform(cfg, is_train=True)
    tfm_test = build_transform(cfg, is_train=False)
    
    train_loader = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                    data_source=dataset.train_x,
                    batch_size=args.batch_size,
                    tfm=tfm_train,
                    is_train=True,
                    dataset_wrapper=None
                )
    
    test_loader = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.test,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    # drop_last=False,
                    dataset_wrapper=None
                )
    
    val_loader = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.val,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    # drop_last=False,
                    dataset_wrapper=None
                )


    run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()