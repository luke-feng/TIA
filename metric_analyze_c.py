import pandas as pd
import os
import glob
import logging
import os
import sys
from DFL_util import dataset_model_set, create_graph_object
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset, DataLoader
from typing import OrderedDict, List, Optional
import networkx as nx
from LP_Cluster import LP_Cluster_Algo


'''This script is used to do the topo inference through clustering method.'''

    
def main():
    TOPOLOGY = ["star", "ring", "ER_0.3", "ER_0.5", "ER_0.7"]
    ROUND = [10, 30]
    NUM_CLIENTS = [20]
    DATASET =  ["Mnist", "FMnist","Cifar10","Cifar10no"] #,
    MODEL = ["mlp", "mobile"]
    IID = [1]
    MAX_EPOCHS = [10]  # fixed as 10
    SEED = [42]
    ALPHA = 0.1
    
    METRIC = ["Average_Loss","Cosine_Similarity", "Euclidean_Distance"] #, "Average_Entropy"
    
    ALGO = ["Algo_row", "Algo_average", "Algo_multi", "Algo_column"]
    
    ClUSTERING = ["Kmeans", "GMM", "Spectral"]
    
    case_dict = {}
    
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
                                
                                print(file_name)
                                
                                case_dict["File_name"] = file_name
                                
                                G = create_graph_object(num, topo)
                                                     
                                for metric in METRIC:
                                    case_dict["Flag"] = metric in ["Cosine_Similarity", "Euclidean_Distance"]
                                    case_dict["Metric"] = metric
                                    
                                    for algo in ALGO:
                                        case_dict["Algo"] = algo
                                        
                                        for clustering in ClUSTERING:
                                            case_dict["Clustering"] = clustering
                                            
                                            for rou in ROUND:
                                            
                                                topo_algo = LP_Cluster_Algo(case_dict=case_dict, graph_object=G, excel_path=f"./saved_results/{metric}/Extra_cases_{num}/{file_name}",
                                                                           attack_path=f"./saved_results/Attack_Performance/Extra_cases_{num}/Link_Prediction_Clustering_{iid}_{rou}                                                                                               .xlsx", 
                                                                           round_case=rou)
                                                
                                                topo_algo.execute_attack()
                                
                                
                                    
                                
                                

                                
                                



if __name__ == '__main__':
    main()
