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


'''This script is used to calculate different node metrics with regard to different DFL scenarios.'''
    

def main():
    TOPOLOGY = ["ER_0.3", "star", "ring", "ER_0.5", "ER_0.7"]#, ] #"fully", , "ER_0.2" 
    ROUND = [9]
    NUM_CLIENTS = [10] # 10
    DATASET =  ["Cifar10", "FMnist", "Mnist", "Cifar10no"] 
    MODEL = ["mlp", "mobile"]
    IID = [0]
    BATCH_SIZE = 128
    SIZE = 2500  # fixed as 2500
    MAX_EPOCHS = [10]  # fixed as 10
    SEED = [42]
    ALPHA = 0.1
    
    METRIC = ["euclidean", "loss", "cosine", "entropy"]
    
    backup_dataset = CIFAR10DatasetNoAugmentation()
    
    for dataset in DATASET:
        for model_name in MODEL:
            # dataset and model setting
            global_dataset, global_model = dataset_model_set(dataset, model_name)
            
            if global_dataset == None or global_model == None:
                continue
                
            for topo in TOPOLOGY:
                                
                for iid in IID:
                    for seed in SEED:
                        for max_epoch in MAX_EPOCHS:
                            for num in NUM_CLIENTS:
                                
                                file_name = dataset + "_" + model_name + "_" + topo + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(seed) + "_" + str(max_epoch)  + "_" + str(num)
                                print(file_name)
                            
                                # separate client's dataset: # A list containing all dataloaders
                                if iid:
                                    if dataset != "Cifar10":
                                        train_subsets, test_subsets = sample_iid_train_data(global_dataset, num, SIZE, seed)
                                    else:
                                        train_subsets, test_subsets = sample_iid_train_data(backup_dataset, num, SIZE, seed)
                                    train_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=False, num_workers=12) for i in train_subsets]
                                    
                                    # test_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=False, num_workers=12) for i in test_subsets]
                                else:
                                    if dataset != "Cifar10":
                                        train_subsets = sample_dirichlet_train_data(global_dataset.train_set, num, SIZE * num,
                                                                                ALPHA, seed)
                                    else:
                                        train_subsets = sample_dirichlet_train_data(backup_dataset.train_set, num, SIZE * num,
                                                                                ALPHA, seed)
                                    train_loaders = [DataLoader(train_subsets[i], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
                                                     for i in range(num)]
                                                     
                                for metric in METRIC:
                                    print(metric)
                                    
                                    if metric in ["loss", "entropy"]:
                                        model_directory = f"./saved_models/Extra_cases_{num}/{file_name}/Local_models"
                                    else:
                                        model_directory = f"./saved_models/Extra_cases_{num}/{file_name}/Aggregated_models"
                                        
                                    for rou in ROUND: # we only consider the last round case now
                                        w_locals = []
                                        
                                        for client_idx in range(num):                           
                                            model_path = model_directory + f"/Round_{rou}/client_{client_idx}.pth"
                                            state_dict = torch.load(model_path)
                                            
                                            w_locals.append(state_dict)
                                            
                                        node_metric = Node_Metrics(model=global_model.to('cuda'), w_locals=w_locals, data_loaders=train_loaders, file_name=file_name,
                                                  dir_name=f'./saved_results/Extra_cases_{num}/', rou=rou, metric=metric)
                                                  
                                        node_metric.execute_attack()
                                    
                                
                                
                                  
                        
                        
        
                                    
                        
                        

if __name__ == '__main__':
    main()