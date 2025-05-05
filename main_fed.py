import torch 
torch.set_float32_matmul_precision("medium")

import logging
import os
import sys
from torch.utils.data import DataLoader
from DFL_util import parse_experiment_file, dataset_model_set, create_nodes_list, \
    create_adjacency, sample_iid_train_data, sample_dirichlet_train_data, create_random_adjacency, \
    save_params, create_graph_object, fed_avg
from lightning import Trainer
from MyCustomCallback import MyCustomCheckpoint
import json
from datetime import datetime
import time
import torch
import networkx as nx
import pickle as pk
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

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
    TOPOLOGY = ["star", "ring","ER_0.3", "ER_0.5", "ER_0.7",'Abilene', 'GÉANT', 'synth50', 'rf1755', 'rf3967', 'atlanta', 'brain', 'cost266', 'dfn-bwin', 'dfn-gwin', 'di-yuan', 'france',  'germany50', 'giul39', 'india35', 'janos-us', 'janos-us-ca', 'newyork', 'nobel-eu', 'nobel-germany', 'nobel-us', 'norway', 'pdh', 'pioro40', 'polska', 'sun', 'ta1', 'ta2', 'zib54']
    # TOPOLOGY = ["ring","ER_0.3", "ER_0.5", "ER_0.7",'Abilene', 'GÉANT', 'synth50', 'rf1755', 'rf3967', 'atlanta', 'cost266', 'dfn-bwin', 'dfn-gwin', 'di-yuan', 'france',  'germany50', 'giul39', 'india35', 'janos-us', 'janos-us-ca', 'newyork', 'nobel-eu', 'nobel-germany', 'nobel-us', 'norway', 'pdh', 'pioro40', 'polska', 'sun', 'ta1', 'ta2', 'zib54']
    # TOPOLOGY = ["ring"]
    ROUND = 20
    NUM_CLIENTS = [10, 20,30,12,22,50,79,87, 15, 161, 37, 11, 25, 35, 26, 39, 16, 28, 17, 14,  40,  27, 24, 65, 54]
    # NUM_CLIENTS = [10]
    # DATASET = ["Cifar10no", "Cifar10", "Mnist","FMnist", "imagenet100", "pcam", "svhn"]
    DATASET = ["imagenet10"]
    # DATASET = ["FMnist"]
    # MODEL = ["mlp", "mobile"， "resnet","pf"]
    MODEL = ["pf"]
    IID = [1]
    BATCH_SIZE = 512
    SIZE = 1250  # fixed as 2500 to 10 clients, 1250 to 20 clients and 834 to 30 clients
    # MAX_EPOCHS = [3, 10]
    MAX_EPOCHS = [3]
    SEED = [42]
    ALPHA = 0.1
    
    cur_dir = os.getcwd()
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
                                ROUND =  num
                                
                                # G = create_graph_object(num, topo)
                                toponame_file = f"{cur_dir}/topologies/{num}_{topo}.pk"
                                if os.path.exists(toponame_file):  
                                    with open(toponame_file, "rb") as f:
                                        G = pk.load(f)
                                else:
                                    continue
                                                           
                                
                                
                                file_name = dataset + "_" + model_name + "_" + topo + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(seed) + "_" + str(max_epoch) + "_" + str(num)
                                print(file_name)
                                # Directory and log file for model results logging
                                log_directory = f'{cur_dir}/saved_logs/Extra_cases/{file_name}'
                                os.makedirs(log_directory, exist_ok=True)
                                model_directory = f'{cur_dir}/saved_models/Extra_cases/{file_name}'
                                try:
                                    os.makedirs(model_directory)
                                except:
                                    continue
                                
                                # model_log_file = os.path.join(log_directory, 'fed_model_result.log')
                                # model_logger = setup_logger('model_logger_{file_name}', model_log_file)
                            
                                # separate client's dataset: # A list containing all dataloaders
                                if iid:
                                    train_subsets, test_subsets, train_subsets_indices, test_subsets_indices = sample_iid_train_data(global_dataset, num, SIZE, seed)
                                    train_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for i in train_subsets]
                                    
                                    test_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) for i in test_subsets]
                                else:
                                    print("non-iid")
                                    train_subsets, test_subsets, train_subsets_indices, test_subsets_indices = sample_dirichlet_train_data(global_dataset, num, SIZE * num,
                                                                                ALPHA, seed)
                                    train_loaders = [DataLoader(train_subsets[i], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
                                                     for i in range(num)]
                                                     
                                    # test_loaders = [0]*num
                                    test_loaders = [DataLoader(i, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) for i in test_subsets]
                                print("dataloader created+++++++++++++++++++++++++++")
                                # nodes setting
                                # nodes_list = create_nodes_list(num, [train_loaders, test_loaders], global_model, file_name)
                                
                                
                                node_adj_list = create_adjacency(range(num), G)
                            
                                # trainer setting
                                start_time = time.time()
                                # print(f"Training start: {start_time}")
                                
                                try:
                                    # save dataloader for future attack
                                    train_loaders_file = os.path.join(model_directory, 'train_loaders.pk')
                                    with open(train_loaders_file, "wb") as trl:
                                        pk.dump(train_loaders, trl)
                                        
                                    test_loaders_file = os.path.join(model_directory, 'test_loaders.pk')
                                    with open(test_loaders_file, "wb") as tsl:
                                        pk.dump(test_loaders, tsl)
                                    
                                    
                                except Exception as e:
                                    print(e)
                                
                                train_subsets_file = os.path.join(model_directory, 'train_subsets_file.pk')
                                with open(train_subsets_file, "wb") as f:
                                    pk.dump(train_subsets_indices, f)
                                    
                                test_subsets_file = os.path.join(model_directory, 'test_subsets_file.pk')
                                with open(test_subsets_file, "wb") as f:
                                    pk.dump(test_subsets_indices, f)
                                
                                node_adj_list_file = os.path.join(model_directory, 'node_adj_list.pk')
                                with open(node_adj_list_file, "wb") as ajd:
                                    pk.dump(node_adj_list, ajd)
                                    
                                    
                                # save initial params to filesyststem for large scale network
                                for client in node_adj_list:
                                    r = 0 - 1
                                    init_node_file_path = f"{model_directory}/Aggregated_models/Round_{r}"
                                    os.makedirs(init_node_file_path, exist_ok=True)
                                    
                                    save_params(global_model.state_dict(), round_num=r, file_name=init_node_file_path
                                                    , client_id=client, is_global=True)
                                    
                                    
                                # start dfl training
                                for r in range(ROUND):
                                    # model_logger.info(f"{r} round start:")
                                    last_round_file_path = f"{model_directory}/Aggregated_models/Round_{r-1}/"
                                    print(f"Training start for round: {r}")
                                    # training process
                                    for node_id in range(num):
                                        print(f"Training start for node: {node_id}")
                                        client_path = last_round_file_path + f"client_{node_id}.pth"
                                        
                                        # load last round params
                                        if os.path.exists(client_path):                                        
                                            last_round_params = torch.load(client_path)
                                        
                                            client = global_model
                                            client.load_state_dict(last_round_params)
                                                                                           
                                        else:
                                            print("error! last round file does not exist!!!")
                                            client = None
                                        
                                        local_trainer = Trainer(max_epochs=max_epoch, accelerator="gpu", devices=1, logger=False,
                                                        # callbacks = [MyCustomCheckpoint(save_dir=f"{model_directory}/Local_models/Round_{r}",
                                                        # idx=client.idx, rou=r, logger=model_logger)],
                                                        enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False)
                                        
                                        # print("trainer created++++++++++++++")
                                        train_loader = train_loaders[node_id]
                                        # local_trainer.fit(client.model, client.train_loader)
                                        local_trainer.fit(client, train_loader)
                                                             
                                        # save model params
                                        save_params(client.state_dict(), round_num=r, file_name=f"{model_directory}/Local_models/Round_{r}"
                                                    , client_id=node_id, is_global=True)
                                        
                                        # current client local training finished
                                        local_trainer = None
                                        client = None
                           
                                    # Aggregation process
                                    nodes_list = {}
                                    # load clients from filesystem
                                    for node_id in range(num):
                                        # print(f"Agg. start for Node: {node_id}")
                                        client_path = f"{model_directory}/Local_models/Round_{r}/client_{node_id}.pth"
                                        if os.path.exists(client_path):
                                            client_params = torch.load(client_path)
                                            nodes_list[node_id] = client_params
                                    
                                    # fedavg
                                    for node_id in nodes_list:
                                        client_params = nodes_list[node_id]
                                        agged_client_params = fed_avg(client_params, nodes_list, node_adj_list[node_id])
                                        save_params(agged_client_params, round_num=r, file_name=f"{model_directory}/Aggregated_models/Round_{r}"
                                                    , client_id=node_id, is_global=True)
                                        
                                        if r == ROUND-1:
                                            local_trainer = Trainer(accelerator="gpu", devices=1, logger=False,
                                                            # callbacks = [MyCustomCheckpoint(save_dir=f"{model_directory}/Local_models/Round_{r}",
                                                            # idx=client.idx, rou=r, logger=model_logger)],
                                                            enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=True)
                                            client = global_model
                                            client.load_state_dict(agged_client_params)
                                            test_loader = test_loaders[node_id]
                                            local_trainer.test(client, test_loader)
                                            local_trainer = None
                                        
                                            client = None
                                    nodes_list = None
                                        
                        
                                end_time = time.time()
                                print(f"Finished in {end_time-start_time} seconds")


if __name__ == '__main__':
    main()
