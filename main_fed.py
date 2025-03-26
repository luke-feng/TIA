import logging
import os
import sys
from torch.utils.data import DataLoader
from DFL_util import parse_experiment_file, dataset_model_set, create_nodes_list, \
    create_adjacency, sample_iid_train_data, sample_dirichlet_train_data, create_random_adjacency, \
    save_params, create_graph_object
from lightning import Trainer
from MyCustomCallback import MyCustomCheckpoint
import json
from datetime import datetime
import time
import torch
import networkx as nx


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    handler.setFormatter(formatter)

    return logger
    

'''This script is used to batch run model files for DFL topo attack experiments.''' 


def main():
    TOPOLOGY = ["star", "ring","ER_0.3", "ER_0.5", "ER_0.7"]
    ROUND = 10
    NUM_CLIENTS = [10]
    DATASET = ["Cifar10no", "Cifar10", "Mnist","FMnist"]
    MODEL = ["mlp", "mobile"]
    IID = [0]
    BATCH_SIZE = 128
    SIZE = 2500  # fixed as 2500 to 10 clients, 1250 to 20 clients and 834 to 30 clients
    MAX_EPOCHS = [3, 10] 
    SEED = [42]
    ALPHA = 0.1
    
    
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
                                file_name = dataset + "_" + model_name + "_" + topo + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(seed) + "_" + str(max_epoch) + "_" + str(num)
                                
                                # Directory and log file for model results logging
                                model_directory = f'./saved_logs/Extra_cases/{file_name}'
                                os.makedirs(model_directory, exist_ok=True)
                                model_log_file = os.path.join(model_directory, 'fed_model_result.log')
                                model_logger = setup_logger('model_logger_{file_name}', model_log_file)
                            
                                # separate client's dataset: # A list containing all dataloaders
                                if iid:
                                    train_subsets, test_subsets = sample_iid_train_data(global_dataset, num, SIZE, seed)
                                    train_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=True, num_workers=12) for i in train_subsets]
                                    
                                    test_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=False, num_workers=12) for i in test_subsets]
                                else:
                                    print("non-iid")
                                    train_subsets = sample_dirichlet_train_data(global_dataset.train_set, num, SIZE * num,
                                                                                ALPHA, seed)
                                    train_loaders = [DataLoader(train_subsets[i], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
                                                     for i in range(num)]
                                                     
                                    test_loaders = [0]*num
                            
                                # nodes setting
                                nodes_list = create_nodes_list(num, [train_loaders, test_loaders], global_model, file_name)
                                
                                G = create_graph_object(num, topo)
                                
                                nodes_list = create_adjacency(nodes_list, G)
                            
                                # trainer setting
                                start_time = time.time()
                                for r in range(ROUND):
                                    model_logger.info(f"{r} round start:")
                                    
                                    # training process
                                    for client in nodes_list:
                                                                
                                        local_trainer = Trainer(max_epochs=max_epoch, accelerator="auto", devices="auto", logger=False,
                                                        callbacks = [MyCustomCheckpoint(save_dir=f"saved_models/Extra_cases_{num}/{file_name}/Local_models/Round_{r}",
                                                        idx=client.idx, rou=r, logger=model_logger)],
                                                        enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False)
                                                                
                                        local_trainer.fit(client.model, client.train_loader)
                                        client.set_current_params(client.model.state_dict())  # store the current trained model params
                                        
                                
                                    # Aggregation process
                                    for client in nodes_list:
                                        client.aggregate_weights()
                                        
                                        save_params(client.aggregated_params, round_num=r, file_name=f"saved_models/Extra_cases_{num}/{file_name}/Aggregated_models/Round_{r}"
                                                    , client_id=client.idx, is_global=True)
                        
                                end_time = time.time()
                                model_logger.info(f"Finished in {end_time-start_time} seconds")


if __name__ == '__main__':
    main()
