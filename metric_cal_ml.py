import logging
import os
import sys
from DFL_util import parse_experiment_file, dataset_model_set, create_nodes_list, \
    create_adjacency, sample_iid_train_data, sample_dirichlet_train_data, create_adjacency, \
    save_params
from Node_Metrics import Node_Metrics
from Dataset.cifar10 import CIFAR10Dataset, CIFAR10DatasetNoAugmentation, CIFAR10DatasetExtendedAugmentation
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset, DataLoader

    
def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    handler.setFormatter(formatter)

    return logger
    

'''This script is used to calculate different node metrics with regard to machine learning scenarios.'''


def main():
    NUM_CLIENTS = [10] # 10
    DATASET =  ["Cifar10"]#no", "Mnist", "FMnist"]
    MODEL = ["mlp", "mobile"]
    BATCH_SIZE = 128
    SIZE = 2500  # fixed as 2500
    MAX_EPOCHS = [100]  # fixed as 10
    SEED = [42]
    
    METRIC = ["loss", "cosine", "entropy", "euclidean"]
    
    backup_dataset = CIFAR10DatasetNoAugmentation()
    
    for dataset in DATASET:
        for model_name in MODEL:
            # dataset and model setting
            global_dataset, global_model = dataset_model_set(dataset, model_name)
            
            if global_dataset == None or global_model == None:
                continue

            for seed in SEED:
                for max_epoch in MAX_EPOCHS:
                    for num in NUM_CLIENTS:
                        file_name = dataset + "_" + model_name + str(seed) + "_" + str(max_epoch) + "_" + str(num) + "_" + "2500"
                    
                        if dataset != "Cifar10":
                            train_subsets, test_subsets = sample_iid_train_data(global_dataset, num, SIZE, seed)
                        else:
                            train_subsets, test_subsets = sample_iid_train_data(backup_dataset, num, SIZE, seed)
                            
                        train_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=False, num_workers=12) for i in train_subsets]
                        
                        # test_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=False, num_workers=12) for i in test_subsets]
                        
                        model_directory = f"./saved_models/Extra_cases_ML/{file_name}"
                              
                        for metric in METRIC:
                                
                            for epoch in [99]: # we only consider the last round case now
                                w_locals = []
                                
                                for client_idx in range(num):                           
                                    model_path = model_directory + f"/Client_{client_idx}/model-epoch={epoch}.pth"
                                    state_dict = torch.load(model_path)
                                    
                                    w_locals.append(state_dict)
                                    
                                node_metric = Node_Metrics(model=global_model.to('cuda'), w_locals=w_locals, data_loaders=train_loaders, file_name=file_name, 
                                          dir_name=f'./saved_results/Extra_cases_ML/', rou=epoch, metric=metric)
                                          
                                node_metric.execute_attack()
                                    
                                
                                
                                  
                        
                        
        
                                    
                        
                        

if __name__ == '__main__':
    main()