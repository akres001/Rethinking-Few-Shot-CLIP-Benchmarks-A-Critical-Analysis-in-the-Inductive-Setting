import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, dataset

from typing import Dict, List

from FSL.clip import clip

from tqdm import tqdm
import copy



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
    
    "ImageNetDogs": "a photo of a {}.",
    "ImageNetNoDogs": "a photo of a {}.",
    
    "ImageNetDogs": "a photo of a {}.",
    "ImageNetNoDogs": "a photo of a {}.",
}

class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )
 
    
    def calc_importance(self, dataloaders: DataLoader, classnames: Dict[str, torch.Tensor], aggregate=True, extra_text="", feature_loss=False) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances_all = []
        for ds in dataloaders:
            importances = self.zerolike_params_dict(self.model)
            
            dataloader = dataloaders[ds]
            ds_name = dataloader.dataset.cfg.DATASET.NAME
            
            print("Current ds", ds)
            print("len dataloader", len(dataloader))
            print("DS NAME", ds_name)
            
            ds_name = ds
            
            text_weights = []
            for classname in classnames[ds_name]:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                texts = [t.format(classname) for t in [CUSTOM_TEMPLATES[ds_name]]]
                texts = clip.tokenize(texts).to(self.device)

                text_weights.append(texts)

            text_weights = torch.cat(text_weights)


            for batch in tqdm(dataloader):
                x, labels = batch['img'], batch['label']
                x, labels = x.to(self.device), labels.to(self.device).long()
                self.opt.zero_grad()
                logits_per_image, _ = self.model(x, text_weights)
                loss = criterion(logits_per_image, labels)

                    
                loss.backward()

                for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
                ):
                    if p.grad is not None:
                        imp.data += p.grad.data.clone().pow(2)#.cpu()

            # average over mini batch length
            for k, imp in importances.items():
                imp.data /= float(len(dataloader))
            
            importances_all.append(copy.deepcopy({k : imp.cpu() for k, imp in importances.items()}))
            torch.save(copy.deepcopy({k : imp.cpu() for k, imp in importances.items()}), f"ssd_importances/importance_{ds}_{extra_text}.pt")
        
        if aggregate:
            importances_out = self.zerolike_params_dict(self.model)
            n_ds = float(len(importances_all))
            print(n_ds)
            for imp_dict in importances_all:
                for n, p in imp_dict.items():
                    importances_out[n] += p / n_ds

            return importances_out
        
        else:
            return {ds : data for ds, data in zip(dataloaders.keys(), importances_all)}
            
    
    def calc_importance_loaded(self, path_importances, ds_list):
        all_importances = torch.load(path_importances)
        all_importances = {k : all_importances[k] for k in all_importances if k in ds_list}
        
        n_ds = float(len(ds_list))
        importances_out = self.zerolike_params_dict(self.model)
        print("len n_ds", n_ds)
        for k in all_importances:
            imp_dict = all_importances[k]
            for n, p in imp_dict.items():
                importances_out[n] += p.to(self.device) / n_ds
        
        return importances_out
    
    
    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
        ignore_params = []
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                if n in ignore_params: continue
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
