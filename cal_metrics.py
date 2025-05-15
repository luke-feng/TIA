import os, sys
import torch
import pickle as pk
from attack_metrics import _curvature_metric, _cosine_metric, _curvature_divergence, \
    _entropy_metric, _euclidean_metric, _jacobian_metric, _loss_metric, _hessian_metric, evaluate_model_metrics
import pandas as pd
import networkx as nx
import copy
from Dataset.mnist import MNISTDataset
from Dataset.cifar10 import CIFAR10Dataset, CIFAR10DatasetNoAugmentation
from Dataset.fmnist import FashionMNISTDataset
from Model.mlp import MLP
from Model.fashionmlp import FashionMNISTModelMLP
from Model.simplemobilenet import SimpleMobileNet
from Model.svhnresnet9 import SVHNResNet9
from Model.pcamrestnet9 import PCAMResNet9
from Model.imagenet10poolformer import ImageNet10PoolFormerS12
import re


TOPOLOGY = ["star", "ring","ER_0.3", "ER_0.5", "ER_0.7",'Abilene', 'GÉANT', 'synth50', 'rf1755', 'rf3967', 'atlanta',  'cost266', 'dfn-bwin', 'dfn-gwin', 'di-yuan', 'france',  'germany50', 'giul39', 'india35', 'janos-us', 'janos-us-ca', 'newyork', 'nobel-eu', 'nobel-germany', 'nobel-us', 'norway', 'pdh', 'pioro40', 'polska', 'sun', 'ta1', 'ta2', 'zib54']
NUM_CLIENTS = [10, 20,30,12,22,50,79,87, 15, 161, 37, 11, 25, 35, 26, 39, 16, 28, 17, 14,  40,  27, 24, 65, 54]

# DATASET = ["Cifar10no", "Cifar10", "Mnist","FMnist", "imagenet100", "pcam", "svhn"]
DATASET = [ "Cifar10"]
# MODEL = ["mlp", "mobile"， "resnet","pf"]
MODEL = ["mobile"]
IID = [1, 0]
MAX_EPOCHS = [3]
ALPHA = 0.1
SEED = [42]
device = 'cuda'

def _get_model(dataset, model_name):
    if dataset == 'Mnist' and model_name == 'mlp':
        return MLP()
    if dataset == 'FMnist' and model_name == 'mlp':
        return FashionMNISTModelMLP()
    if dataset == 'Cifar10' and model_name == 'mobile':
        return SimpleMobileNet()
    if dataset == 'Cifar10no' and model_name == 'mobile':
        return SimpleMobileNet()
    if dataset == 'svhn' and model_name == 'resnet':
        return SVHNResNet9()
    if dataset == 'pcam' and model_name == 'resnet':
        return PCAMResNet9()
    if dataset == 'imagenet10' and model_name == 'pf':
        return ImageNet10PoolFormerS12()


def get_last_round(base_dir):
    round_dirs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("Round_")]

    round_tuples = []
    for d in round_dirs:
        match = re.match(r"Round_(-?\d+)", d)
        if match:
            round_tuples.append((int(match.group(1)), d))


    round_tuples.sort(key=lambda x: x[0])


    if len(round_tuples) >= 2:
        second_last = round_tuples[-2][1]
        last = round_tuples[-1][1]
        return last, second_last
    else:
        return None
    
   

cur_dir = os.getcwd()
topology_path = f"{cur_dir}/topologies/"
saved_model_path =  f"{cur_dir}\\saved_models\\"
all_scenarios = os.listdir(saved_model_path)
metrics_path = f"{cur_dir}\\saved_metrics\\"

for dataset in DATASET:
        for model_name in MODEL:
            # dataset and model setting
            for topo in TOPOLOGY:
               for iid in IID:
                    for seed in SEED:
                        for max_epoch in MAX_EPOCHS:
                            for num in NUM_CLIENTS:
                                ROUND =  num
                                toponame_file = f"{topology_path}/{num}_{topo}.pk"
                                if os.path.exists(toponame_file):  
                                    with open(toponame_file, "rb") as f:
                                        G = pk.load(f)
                                else:
                                    continue
                                
                                scenario_name = dataset + "_" + model_name + "_" + topo + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(seed) + "_" + str(max_epoch) + "_" + str(num)
                                
                                
                                if scenario_name in all_scenarios:
                                    # print(scenario_name)
                                    try:
                                        saved_metrics_path = metrics_path + scenario_name
                                        os.makedirs(saved_metrics_path)
                                    except:
                                        continue
                                    
                                    scenario_folder = saved_model_path + scenario_name
                                    train_loader_file = os.path.join(scenario_folder, "train_loaders.pk")
                                    with open(train_loader_file, 'rb') as f:
                                        train_loaders = pk.load(f)

                                    model = _get_model(dataset, model_name)
                                    try:
                                        agg_ronnds = f'{scenario_folder}\\Aggregated_models\\'
                                        last, second_last = get_last_round(agg_ronnds)
                                        if last==None:
                                            continue
                                        model_folder = f'{scenario_folder}\\Aggregated_models\\{last}\\'
                                        clients = []
                                        for node_id in range(num):
                                            # print(f"load model for node: {node_id}")
                                            client_path = model_folder + f"client_{node_id}.pth"

                                            if os.path.exists(client_path):                                        
                                                last_round_params = torch.load(client_path)
                                            
                                                client = copy.deepcopy(model)
                                                client.load_state_dict(last_round_params)
                                                clients.append(client)
                                                
                                        print(len(clients))       
                                        model_folder = f'{scenario_folder}\\Aggregated_models\\{second_last}\\'
                                        clients_last = []
                                        for node_id in range(num):
                                            # print(f"load last round model for node: {node_id}")
                                            client_path = model_folder + f"client_{node_id}.pth"

                                            if os.path.exists(client_path):                                        
                                                last_round_params = torch.load(client_path)
                                            
                                                client = copy.deepcopy(model)
                                                client.load_state_dict(last_round_params)
                                                clients_last.append(client)
                                        # if len(clients) < num or clients_last < num:
                                        #     continue
                                    except:
                                        print(scenario_name)
                                        continue
                                    
                                    
                                    # for approach in ['_cosine_metric', '_curvature_divergence', '_entropy_metric', '_euclidean_metric', '_jacobian_metric', '_loss_metric']:
                                    # for approach in ['_curvature_metric', '_hessian_metric']:
                                    metrics = {}
                                    metrics['_cosine_metric'] = []
                                    metrics['_curvature_divergence'] = []
                                    metrics['_entropy_metric'] = []
                                    metrics['_euclidean_metric'] = []
                                    metrics['_jacobian_metric'] = []
                                    metrics['_loss_metric'] = []
                                    
                                    for i in range(0, num):
                                        cm_row = []
                                        em_row = []
                                        cd_row = []
                                        lm_row = []
                                        enm_row = []
                                        jm_row = []
                                        
                                        for j in range(0, num):
                                            clients[i].to(device)
                                            clients_last[i].to(device)
                                            clients[j].to(device)
                                            clients_last[j].to(device)

                                            cm = _cosine_metric(clients[i].state_dict(), clients[j].state_dict())
                                            em =_euclidean_metric(clients[i].state_dict(), clients[j].state_dict())
                                            cd =_curvature_divergence(clients[i].state_dict(), clients_last[i].state_dict(), clients[j].state_dict(), clients_last[j].state_dict())
                                            lm, enm, jm = evaluate_model_metrics(model, clients[j].state_dict(), train_loaders[i], max_batches=1)
                                            cm_row.append(cm)
                                            em_row.append(em)
                                            cd_row.append(cd)
                                            lm_row.append(lm)
                                            enm_row.append(enm)
                                            jm_row.append(jm)
                                        
                                        metrics['_cosine_metric'].append(cm_row)
                                        metrics['_curvature_divergence'].append(cd_row)
                                        metrics['_entropy_metric'].append(enm_row)
                                        metrics['_euclidean_metric'].append(em_row)
                                        metrics['_jacobian_metric'].append(jm_row)
                                        metrics['_loss_metric'].append(lm_row)
                                    
                                    for approach in metrics:
                                        df = pd.DataFrame(metrics[approach])
                                        metircs_file = saved_metrics_path + '\\' + approach + '.csv'
                                        df.to_csv(metircs_file)    
                                    
                                    # for approach in ['_cosine_metric', '_curvature_divergence', '_euclidean_metric', 'model_metrics']:
                                    #     metrics = []
                                    #     for i in range(0, num):
                                    #         row = []
                                    #         for j in range(0, num):
                                    #             clients[i].to(device)
                                    #             clients_last[i].to(device)
                                    #             clients[j].to(device)
                                    #             clients_last[j].to(device)
                                    #             if approach == '_curvature_metric':
                                    #                 m = _curvature_metric(model, clients[j].state_dict(),clients_last[j].state_dict(), train_loaders[i], max_batches=1, approach='approx')
                                    #             if approach == '_cosine_metric':
                                    #                 m =_cosine_metric(clients[i].state_dict(), clients[j].state_dict())
                                    #             if approach == '_euclidean_metric':
                                    #                 m =_euclidean_metric(clients[i].state_dict(), clients[j].state_dict())
                                                    
                                    #             if approach == '_hessian_metric':
                                    #                 m = _hessian_metric(model, clients[j].state_dict(), train_loaders[i], max_batches=1)
                                                
                                    #             if approach == '_jacobian_metric':
                                    #                 m = _jacobian_metric(model, clients[j].state_dict(), train_loaders[i], max_batches=1)
                                                    
                                    #             if approach == '_loss_metric':
                                    #                 m = _loss_metric(model, clients[j].state_dict(), train_loaders[i], max_batches=1)
                                                    
                                    #             if approach == '_entropy_metric':
                                    #                 m =_entropy_metric(model, clients[j].state_dict(), train_loaders[i], max_batches=1)               
                                                
                                    #             if approach == '_curvature_divergence':
                                    #                 m =_curvature_divergence(clients[i].state_dict(), clients_last[i].state_dict(), clients[j].state_dict(), clients_last[j].state_dict())
                                                
                                    #             row.append(m)
                                    #         metrics.append(row)
                                            
                                    #     df = pd.DataFrame(metrics)
                                    #     metircs_file = saved_metrics_path + '\\' + approach + '.csv'
                                    #     df.to_csv(metircs_file)

