from subprocess import Popen
import time
import os
from fnmatch import fnmatch
import torch
import numpy as np
import argparse

torch.set_num_threads(5)

def get_available_memory(dev):
    return torch.cuda.mem_get_info(dev)[0] // 1024 ** 2

def select_free_device():
    list_memories = []
    for i in range(torch.cuda.device_count()):
        list_memories.append(get_available_memory(i))
    return str(np.argmax(list_memories))

def main(args):
    
    if args.models == 'all':
        models = ['sepres', 'lora', 'tcp', 'prograd', 'kgcoop', 'sep', 'coprompt', 'maple', 'coop', 'adapt', 'promptsrc', 'taskres', 'cocoop', 'ivlp']
    else:
        models = args.models.split(",")
    
    # models = ['sepres']
              
    ATTEMPT = str(args.attempt)
    MODEL_TYPE = args.model_type
    DATASETS = args.datasets
    SOURCEMODELS = args.source_model
    USE_FORGET_MODEL = '0' if SOURCEMODELS == '' else '1'

    # List of CUDA devices available for usage
    cuda_devices = ['0', '1', '2', '3', '4', '5', '6', '7']  


    print("ATTEMPT", ATTEMPT, "MODEL_TYPE", MODEL_TYPE, "USE_FORGET_MODEL", USE_FORGET_MODEL, "DATASETS", DATASETS, "SOURCEMODELS", SOURCEMODELS)
    for ii, model in enumerate(models):


        cuda_device = select_free_device()
        memory = get_available_memory(int(cuda_device))

        while memory < 15000:
            print("no available memory, retry")      
            # retry in 5 min 
            time.sleep(5 * 60)

            cuda_device = select_free_device()
            memory = get_available_memory(int(cuda_device))


        print(f"Starting {model} on {cuda_device} with memory {memory}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_device

        Popen(["bash", f"run_{model}.sh", MODEL_TYPE, ATTEMPT, USE_FORGET_MODEL, DATASETS, SOURCEMODELS],
                         stdout=open(f'../few_shot_out/results_{ATTEMPT}/null_{model}_type_{MODEL_TYPE}_forget_{USE_FORGET_MODEL}', 'w'),
                         stderr=open(f'../few_shot_out/results_{ATTEMPT}/logfile_{model}_type_{MODEL_TYPE}_{USE_FORGET_MODEL}.log', 'w'),
                         start_new_session=True,
                         shell=False,
                         env=env  # Pass the modified environment
                         )

        # allow some time between each process
        # 60 min for now. 
        time.sleep(5 * 60)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=str, required=True)
    parser.add_argument("--datasets", default=',', required=True, type=str, help='stanford_dogs,caltech101,')
    parser.add_argument("--model_type", default='vit_b16', type=str)
    parser.add_argument("--source_model", default='', type=str)
    parser.add_argument("--models", default='all', type=str)
    
    
    args = parser.parse_args()
    
    
    os.makedirs(f"../few_shot_out/results_{args.attempt}", exist_ok=True)

    main(args)
    
