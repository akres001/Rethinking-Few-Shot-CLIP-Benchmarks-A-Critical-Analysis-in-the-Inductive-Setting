import torch
import torch.nn.functional as F

from utils import *
from collections import defaultdict
import os
import pickle

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

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


def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        # for i, (images, target) in enumerate(loader):
        for i, (batch) in enumerate(tqdm(loader)):
            images, target = batch['img'], batch['label']
            
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = False
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        # for i, (images, target) in enumerate(tqdm(train_loader)):
        for i, (batch) in enumerate(tqdm(train_loader)):
            images, target = batch['img'], batch['label']
            
            template = dataset.template[0]
            # print("Template", template)
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}, count_iters: {:.1f}'.format(current_lr, acc_train, loss_epoch, count_iters))

        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    
    acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    
    if args.save_results:
        if args.source_model:
            idx_path = '1'
        else:
            idx_path = '0'
        
        # we are in FSL 
        path_saved = f'../few_shot_out/results_{args.attempt}/{args.dataset}/CLIPLora_{idx_path}_summary.pkl'
        path_file = f'../few_shot_out/results_{args.attempt}/{args.dataset}/CLIPLora/vit_b16_{args.shots}shots_{idx_path}'
        # print("path_saved", path_saved)
        print(os.getcwd(), path_saved)
        if os.path.exists(path_saved):
            with open(path_saved, "rb") as f:
                results_prev = pickle.load(f)
            
            if path_file in results_prev:
                results_prev[path_file][0][f'seed{args.seed}'] = [acc_test]
            else:
                results_prev[path_file] = [{f'seed{args.seed}' : [acc_test]}]
                
        else:
            results_prev = defaultdict(list)
            results_prev[path_file] = [{f'seed{args.seed}' : [acc_test]}]

        with open(path_saved, "wb") as f:
            pickle.dump(results_prev, f)
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return
            
    
            
