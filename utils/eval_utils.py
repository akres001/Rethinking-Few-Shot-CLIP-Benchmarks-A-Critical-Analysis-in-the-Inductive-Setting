import numpy as np
import torch
from FSL.clip import clip

from PIL import Image
from tqdm import tqdm
import random
from sklearn import metrics

import torchvision.transforms as T
from sklearn.metrics import confusion_matrix


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    
    "PLANTDOC": "a photo of a {}.",
    "CUB": "a photo of a {}, a type of bird.",
    "StanfordDogs": "a photo of a {}.",
}



def weighted_loss(forget_ds:str, val_ds:list, mmd_sim_text:dict, mmd_sim_imgs:dict) -> dict:
    """
    Compute normalized weight scores for validation datasets based on MMD similarity
    with the forget dataset.

    Args:
        forget_ds: Name of the dataset to be forgotten
        val_ds: List of validation dataset names
        mmd_sim_text: Dictionary of MMD similarities between text features
        mmd_sim_imgs: Dictionary of MMD similarities between image features

    """
    # print("forget_ds", forget_ds)
    weights = []
    for k in val_ds:
        key_mmd = '_'.join(sorted([k, forget_ds]))
        weights.append(0.5 * (mmd_sim_text[key_mmd] + mmd_sim_imgs[key_mmd]))
        
    weights = np.array(weights) 
    weights = weights/weights.sum()
    weights_dict = {k : w for w, k in zip(weights, val_ds)}
    
    return weights_dict

def get_model(device: str = 'cpu', arch: str = "ViT-B/16", load_path: str = "")-> torch.nn.Module:
    """
    Load a CLIP model, optionally from a custom checkpoint.

    Args:
        device: Target device ('cpu' or 'cuda')
        arch: CLIP model architecture (e.g., "ViT-B/16")
        load_path: Path to custom model checkpoint (optional)

    """
    backbone_name = arch 
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    print("Loading model..")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict()).float().to(device).eval()
    if load_path:
        print(f"LOADING FROM {load_path}") 
        model.load_state_dict(torch.load(load_path, map_location="cpu")['model_dict'])
        model = model.float().to(device).eval()
            
    return model


def eval_all_ds(model : torch.nn.Module, datasets_cls : dict, all_loaders : dict, device : str='cpu') -> dict:
    
    """
    Evaluate a CLIP model on multiple datasets using zero-shot classification.
    
    Args:
        model: CLIP model to evaluate
        datasets_cls: Dictionary mapping dataset names to their class information
        all_loaders: Dictionary mapping dataset names to their data loaders
        device: Device to run evaluation on ('cpu' or 'cuda')
    
    """
        
    results = {ds: {} for ds in all_loaders}
    for ds in all_loaders:
        model.eval()
        test_loader = all_loaders[ds]
        
        classnames = datasets_cls[ds].classnames
        clip_weights = clip_classifier(classnames, [CUSTOM_TEMPLATES[ds]], model).to(device)

        acc = evaluate_clip_zs(model, test_loader, clip_weights, device=device, out_conf=False)
        results[ds]['all'] = {'all_ds' : acc}     
        print(f"{10*'+++'} {ds} - {acc} {10*'+++'}")
    
    return results



def clip_classifier(classnames: list, template: list, clip_model: torch.nn.Module) -> torch.Tensor:
    """
    Generate CLIP classifier weights for given classnames.

    Args:
        classnames (list): List of class names.
        template (list): List of prompt templates.
        clip_model (torch.nn.Module): Pre-trained CLIP model.

    """
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(clip_model.visual.conv1.weight.device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1)#.cuda()
    return clip_weights


def cls_acc(output: np.ndarray, target: np.ndarray, topk: int = 1) -> float:
    """
    Compute classification accuracy.

    Args:
        output (np.ndarray): Model output logits.
        target (np.ndarray): True labels.
        topk (int): Number of top predictions to consider.

    """

    pred = np.argmax(output, axis=1)
    
    # Check if predictions match the target
    correct = pred == target.reshape(1, -1)
    
    # Calculate accuracy
    acc = correct[:topk].reshape(-1).sum(0)
    acc = 100 * acc / target.shape[0]
    
    return acc


def evaluate_clip_zs(model : torch.nn.Module, 
                     loader : torch.utils.data.DataLoader, 
                     clip_weights : torch.Tensor, 
                     device : str=None,
                     out_conf : bool=False,
                     output_probs: bool=False,
                     ignore_classes : bool=None,
                     test_ignored : bool=False, 
                     return_logits: bool=False):
    
    """
    Evaluate a CLIP model using zero-shot classification.
    
    Args:
        model: CLIP model to evaluate
        loader: DataLoader providing image-label batches
        clip_weights: Precomputed CLIP weight vectors for each class
        device: Device to run evaluation on
        out_conf: If True, returns confidence scores (logits)
        output_probs: If True, returns class probabilities
        ignore_classes: List of class indices to ignore
        test_ignored: If True, only evaluate on ignored classes
        return_logits: If True, returns raw logits (deprecated, use return_confidences)
    
    """
    
    model.eval()
    features = []
    labels = []
        
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):    
            images = batch['img']
            target = batch['label']

            images, target = images.to(device), target.to(device)
            # image_features, image_features_projected = mymodel.encode_image(images)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            features.append(image_features.cpu())
            labels.append(target.cpu()) 
            
    labels = torch.cat(labels)
    features = torch.cat(features)
    
    if ignore_classes != None:
        if test_ignored:
            mask = torch.isin(labels, ignore_classes)
        else:
            mask = ~torch.isin(labels, ignore_classes)
        labels = labels[mask]
        features = features[mask]
    
    clip_logits_test = 100. * features @ clip_weights.detach().cpu().numpy()
    acc = cls_acc(clip_logits_test.detach().cpu().numpy(), labels.detach().cpu().numpy())
    acc = acc / 100.
    
    if output_probs:
        probs = torch.nn.functional.softmax(clip_logits_test, dim=-1)
    
    if out_conf:
        if output_probs:
            return acc, (labels, clip_logits_test), probs
            
        return acc, (labels, clip_logits_test)
    
    if output_probs:
        return acc, probs

    return acc
            
def mmd_rbf(X : np.ndarray, Y : np.ndarray, gamma: int=1.0) -> float:
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()