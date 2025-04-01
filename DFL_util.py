import copy
import json
import os
from collections import Counter
from Dataset.cifar10 import CIFAR10Dataset, CIFAR10DatasetNoAugmentation, CIFAR10DatasetExtendedAugmentation
from Dataset.mnist import MNISTDataset
from Dataset.fmnist import FashionMNISTDataset
from Dataset.pcam import PCAMdataset
from Dataset.svhn import SVHNdataset
from Dataset.imagenet100 import ImageNet100Dataset

from Model.mlp import MLP
from Model.mnistmlp import MNISTModelMLP
from Model.fashionmlp import FashionMNISTModelMLP
from Model.cnn import CIFAR10ModelCNN
from Model.simplemobilenet import SimpleMobileNet
from Model.pcamresnet18 import PCAMResNet18
from Model.svhnresnet18 import SVHNResNet18
from Model.imagenet100vgg import ImageNet100VGG


from torch.utils.data import Subset, DataLoader, ConcatDataset
from Node import Node
import numpy as np
import torch
from collections import defaultdict
from lightning.pytorch.loggers import CSVLogger
import networkx as nx


def dataset_model_set(dataset, model):
    g_dataset = None
    g_model = None
    if dataset == "Mnist":
        g_dataset = MNISTDataset()
        if model == "mlp":
            g_model = MLP()
            # g_model = MNISTModelMLP()
        elif model == "cnn":
            pass
        else:
            pass
    elif dataset == "FMnist":
        g_dataset = FashionMNISTDataset()
        if model == "mlp":
            g_model = FashionMNISTModelMLP()
        elif model == "cnn":
            pass
        else:
            pass
    elif dataset == "Cifar10":
        g_dataset = CIFAR10Dataset()
        if model == "mlp":
            pass
        elif model == "cnn":
            g_model = CIFAR10ModelCNN()
        elif model == "mobile":
            g_model = SimpleMobileNet()
    elif dataset == "Cifar10no":
        g_dataset = CIFAR10DatasetNoAugmentation()
        if model == "mlp":
            pass
        elif model == "cnn":
            g_model = CIFAR10ModelCNN()
        elif model == "mobile":
            g_model = SimpleMobileNet()
    elif dataset == 'Cifar10extend':
        g_dataset = CIFAR10DatasetExtendedAugmentation()
        if model == "mlp":
            pass
        elif model == "cnn":
            g_model = CIFAR10ModelCNN()

    elif dataset == 'svhn':
        g_dataset = SVHNdataset()
        if model == "resnet":
            g_model = SVHNResNet18()
    elif dataset == 'imagenet100':
        g_dataset = ImageNet100Dataset()
        if model == "vgg":
            g_model = ImageNet100VGG()
    elif dataset == 'pcam':
        g_dataset = PCAMdataset()
        if model == "resnet":
            g_model = PCAMResNet18()
    return g_dataset, g_model
    
    
def parse_experiment_file(file_path):
    experiment_info = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into key and value parts
            parts = line.strip().split(':')
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                # Check if the value is intended to be a list
                if value.startswith('[') and value.endswith(']'):
                    # Parse the list value using json.loads after replacing single quotes with double quotes
                    # This is necessary because json expects double-quoted strings
                    value = json.loads(value.replace("'", '"'))
                experiment_info[key] = value

    return experiment_info
    

def create_nodes_list(num, datamodule, model, file_name):
    node_list = []
    for i in range(0, num):
        # csv_logger = CSVLogger(save_dir=f"saved_logs/{file_name}", name=f"node_{i}")
        node = Node(i, copy.deepcopy(model), datamodule[0][i], datamodule[1][i])
        node_list.append(node)
    return node_list
    
    
def create_graph_object(num, name):
    if name == "fully":
        G = nx.complete_graph(num)
    elif name == "star":
        G = nx.star_graph(num - 1)
    elif name == "ring":
        G = nx.cycle_graph(num)
    elif name == "random":       
        G = nx.cycle_graph(num)
        for combo in graph_info["combo"]:
            G.add_edge(*combo)
    elif name.startswith("ER_"):
        try:
            # Extract probability and seed from the graph_info
            parts = name.split("_")
            prob = float(parts[1])
            G = nx.erdos_renyi_graph(num, prob, seed=45)
        except (IndexError, ValueError):
            raise ValueError("Invalid format for ER graph type. Expected 'ER_prob'")
    else:
        raise ValueError("Not supported topology.")
        
    return G
        

# def create_adjacency(nodes_list, graph):
#     node_map = {node.idx: node for node in nodes_list}

#     for node in nodes_list:
#         graph_neighbors = list(graph.neighbors(node.idx))
#         node.neigh_list = graph_neighbors
#         # node.neigh = {node_map[neighbor_id] for neighbor_id in graph_neighbors}

#     return nodes_list

def create_adjacency(nodes_id_list, graph):
    node_adj_list = {}
    for node in nodes_id_list:
        graph_neighbors = list(graph.neighbors(node))
        node_adj_list[node] = graph_neighbors
        # node.neigh_list = graph_neighbors
        # node.neigh = {node_map[neighbor_id] for neighbor_id in graph_neighbors}

    return node_adj_list
    
    
def create_random_adjacency(nodes_list, combo): # this function aims to add extra links between specific nodes based on one ring graph
    num_nodes = len(nodes_list)

    for i, node in enumerate(nodes_list):
        left_neighbor = nodes_list[(i - 1) % num_nodes]
        right_neighbor = nodes_list[(i + 1) % num_nodes]
        node.neigh = {left_neighbor, right_neighbor}
        
    for i in combo:
        nodes_list[i[0]].neigh.add(nodes_list[i[1]])
        nodes_list[i[1]].neigh.add(nodes_list[i[0]])
    
    return nodes_list
    

def create_custom_adjacency(nodes_list):
    # Nodes 0
    nodes_list[0].neigh.add(nodes_list[4])
    # Nodes 4
    nodes_list[4].neigh.add(nodes_list[0])
    nodes_list[4].neigh.add(nodes_list[5])
    # Nodes 5
    nodes_list[5].neigh.add(nodes_list[4])
    nodes_list[5].neigh.add(nodes_list[3])
    nodes_list[5].neigh.add(nodes_list[8])
    nodes_list[5].neigh.add(nodes_list[9])
    nodes_list[5].neigh.add(nodes_list[2])
    # Nodes 3
    nodes_list[3].neigh.add(nodes_list[5])
    # Nodes 8
    nodes_list[8].neigh.add(nodes_list[5])
    nodes_list[8].neigh.add(nodes_list[9])
    # Nodes 9
    nodes_list[9].neigh.add(nodes_list[8])
    nodes_list[9].neigh.add(nodes_list[5])
    # Nodes 2
    nodes_list[2].neigh.add(nodes_list[5])
    nodes_list[2].neigh.add(nodes_list[6])
    # Nodes 6
    nodes_list[6].neigh.add(nodes_list[2])
    nodes_list[6].neigh.add(nodes_list[1])
    nodes_list[6].neigh.add(nodes_list[7])
    # Nodes 7
    nodes_list[7].neigh.add(nodes_list[6])
    # Nodes 1
    nodes_list[1].neigh.add(nodes_list[6])

    return nodes_list
    

def sample_iid_train_data(dataset, num_client, data_size, seed):
    np.random.seed(seed)
    train_indices = np.arange(len(dataset.train_set))
    test_indices = np.arange(len(dataset.test_set))
    
    data_size = int(round(len(train_indices)/num_client/2)) - 1

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Create train subsets
    train_subsets_indices = [train_indices[i * data_size:(i + 1) * data_size] for i in range(num_client)]
    train_subsets = [Subset(dataset.train_set, indices) for indices in train_subsets_indices]

    # Calculate test subset size and create test subsets
    test_size = int(len(dataset.test_set) / num_client)
    test_subsets_indices = [test_indices[i * test_size:(i + 1) * test_size] for i in range(num_client)]
    test_subsets = [Subset(dataset.test_set, indices) for indices in test_subsets_indices]

    return train_subsets, test_subsets
    

def build_classes_dict(dataset):
    classes_dict = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        classes_dict[label].append(idx)
    return classes_dict


def sample_dirichlet_train_data(dataset, num_client, data_size, alpha, seed):
    np.random.seed(seed)
    train_dataset = dataset.train_set
    all_indices = np.arange(len(train_dataset))
    np.random.shuffle(all_indices)

    data_size = int(round(len(all_indices)/num_client/2)) - 1
    
    subset_indices = all_indices[:data_size]
    subset_dataset = Subset(train_dataset, subset_indices)

    data_classes = build_classes_dict(subset_dataset)
    per_participant_list = defaultdict(list)
    no_classes = len(data_classes.keys())

    for n in range(no_classes):
        current_class_size = len(data_classes[n])  # Use actual size of the current class
        np.random.shuffle(data_classes[n])
        sampled_probabilities = current_class_size * np.random.dirichlet(
            np.array(num_client * [alpha]))
        for user in range(num_client):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]

    for i in per_participant_list:
        actual_indices = [subset_indices[idx] for idx in
                          per_participant_list[i]]  # Map back to original dataset indices
        per_participant_list[i] = Subset(train_dataset, actual_indices)

    for idx, subset in per_participant_list.items():
        label_counter = Counter({label: 0 for label in range(no_classes)})
        for _, label in subset:
            label_counter[label] += 1
        total_samples = sum(label_counter.values())
        
    # Calculate test subset size and create test subsets
    test_indices = np.arange(len(dataset.test_set))
    test_size = int(len(dataset.test_set) / num_client)
    test_subsets_indices = [test_indices[i * test_size:(i + 1) * test_size] for i in range(num_client)]
    test_subsets = [Subset(dataset.test_set, indices) for indices in test_subsets_indices]

    return per_participant_list, test_subsets


def save_params(params_dict, round_num, file_name, client_id=None, is_global=False):
    # if is_global:
    #     if round_num in [9, 14, 19, 24, 29, 34, 39]:
    #         os.makedirs(file_name, exist_ok=True)
    #         model_name = f"client_{client_id}.pth"
            
    #         path = os.path.join(file_name, model_name)
    #         torch.save(params_dict, path)
    # else:
    #     directory = f"saved_models/{file_name}/Nei_aggregated_models/Round_{round_num}"
    #     os.makedirs(directory, exist_ok=True)
    #     filename = f"client_{client_id}.pth"
    
    os.makedirs(file_name, exist_ok=True)
    model_name = f"client_{client_id}.pth"
    path = os.path.join(file_name, model_name)
    torch.save(params_dict, path)
        
        
def prepare_shadow_dataset(global_dataset, size, seed):
    """This function uses the unused train dataset as the shadow test."""
    np.random.seed(seed)
    all_train_indices = np.arange(len(global_dataset.train_set))

    np.random.shuffle(all_train_indices)
    
    if size > 12500:
        shadow_train_indices = all_train_indices[size:2*size] # 25000:50000
        shadow_train_dataset = Subset(global_dataset.train_set, shadow_train_indices)
        
        first_shadow_test_dataset = global_dataset.test_set
        extra_shadow_test_indices = np.random.choice(all_train_indices[:size], size=size - len(global_dataset.test_set),
                                                     replace=False)
                                                     
        extra_shadow_test_dataset = Subset(global_dataset.train_set, extra_shadow_test_indices)
        shadow_test_dataset = ConcatDataset([first_shadow_test_dataset, extra_shadow_test_dataset])
    else:
        shadow_train_indices = all_train_indices[2*size:3*size] 
        shadow_train_dataset = Subset(global_dataset.train_set, shadow_train_indices)
        
        shadow_test_indices = all_train_indices[3*size:4*size]
        shadow_test_dataset = Subset(global_dataset.train_set, shadow_test_indices)

    shadow_train_dataloader = DataLoader(shadow_train_dataset, batch_size=128, shuffle=True, num_workers=12)
    shadow_test_dataloader = DataLoader(shadow_test_dataset, batch_size=128, shuffle=False, num_workers=12)

    return shadow_train_dataloader, shadow_test_dataloader
    
    
def prepare_evaluation_dataset1(dataset, size, num_client, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))

    np.random.shuffle(all_indices)

    subsets_indices = [all_indices[i * size:(i + 1) * size] for i in range(num_client)]
    subsets = [Subset(dataset.train_set, indices) for indices in subsets_indices]
    
    in_eval_loader = [DataLoader(subsets[i], batch_size=128, shuffle=False, num_workers=12)
                         for i in range(num_client)]
                         
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[size*num_client:]),batch_size=128, shuffle=False, num_workers=12) # this is for the 25000 case
    
    return in_eval_loader, out_eval_loader


def prepare_evaluation_dataset2(dataset, size, num_client, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))
    np.random.shuffle(all_indices)

    # Create subsets indices for in-eval loaders
    subsets_indices = [all_indices[i * size:(i + 1) * size] for i in range(num_client)]
    combined_indices = np.concatenate(subsets_indices)
    
    # Create the combined dataset and dataloader for in-eval
    combined_subset = Subset(dataset.train_set, combined_indices)
    combined_in_eval_loader = DataLoader(combined_subset, batch_size=128, shuffle=False, num_workers=12)
    
    # Create the out-eval dataloader for the remaining data
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[size * num_client:]), batch_size=128, shuffle=False, num_workers=12)
    
    return combined_in_eval_loader, out_eval_loader
    

def prepare_dirichlet_eval_data(dataset, num_client, data_size, alpha, seed):
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))
    np.random.shuffle(all_indices)

    subset_indices = all_indices[:data_size]
    subset_dataset = Subset(dataset.train_set, subset_indices)

    data_classes = build_classes_dict(subset_dataset)
    per_participant_list = defaultdict(list)
    no_classes = len(data_classes.keys())

    for n in range(no_classes):
        current_class_size = len(data_classes[n])  # Use actual size of the current class
        np.random.shuffle(data_classes[n])
        sampled_probabilities = current_class_size * np.random.dirichlet(
            np.array(num_client * [alpha]))
        for user in range(num_client):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = data_classes[n][:min(len(data_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), no_imgs):]
            
    node_data = {}
    for i in per_participant_list:
      node_data[i] = len(per_participant_list[i])

    start_end_indices = []

    # Initialize the start index
    start_idx = 0

    # Iterate through the dictionary to calculate the start and end indices
    for key, value in node_data.items():
        end_idx = start_idx + value
        start_end_indices.append((start_idx, end_idx))
        # Update the start index for the next iteration
        start_idx = end_idx
            
    for i in per_participant_list:
        actual_indices = [subset_indices[idx] for idx in
                          per_participant_list[i]]  # Map back to original dataset indices
        per_participant_list[i] = actual_indices
    
    combined_indices = []   
    for i in range(num_client):
        combined_indices.append(per_participant_list[i])
    combined_indices = np.concatenate(combined_indices)
      
    combined_subset = Subset(dataset.train_set, combined_indices)
    combined_in_eval_loader = DataLoader(combined_subset, batch_size=128, shuffle=False, num_workers=12)  
    
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[data_size:2*data_size]), batch_size=128, shuffle=False, num_workers=12) # use non-training samples as out

    return combined_in_eval_loader, out_eval_loader, start_end_indices
    
def prepare_iid_eval_data(dataset, num_client, data_size, seed):
    size = data_size * num_client
    
    np.random.seed(seed)
    all_indices = np.arange(len(dataset.train_set))

    np.random.shuffle(all_indices)
    
    combined_indices = all_indices[:size]
    combined_subset = Subset(dataset.train_set, combined_indices)
    combined_in_eval_loader = DataLoader(combined_subset, batch_size=128, shuffle=False, num_workers=12)
    
    out_eval_loader = DataLoader(Subset(dataset.train_set, all_indices[size:2 * size]), batch_size=128, shuffle=False, num_workers=12) # use non-training samples as out

    return  combined_in_eval_loader, out_eval_loader, [(i * data_size, (i + 1) * data_size) for i in range(num_client)]



def ML_train_test(train_set, test_set, size, seed):
    '''np.random.seed(seed)
    train_indices = np.random.choice(len(train_set), size=size, replace=False)  # no duplicates
    final_train = Subset(train_set, train_indices)

    final_test = Subset(test_set, range(int(size * 0.4)))'''
    
    np.random.seed(seed)
    all_train_indices = np.arange(len(train_set))

    np.random.shuffle(all_train_indices)
    final_train_indices = all_train_indices[:size]
    final_train = Subset(train_set, final_train_indices)
    
    all_test_indices = np.arange(len(test_set))
    
    np.random.shuffle(all_test_indices)
    final_test_indices = all_test_indices[:int(size*0.4)]
    final_test = Subset(test_set, final_test_indices)

    # total_labels = Counter(y for _, y in final_train)
    # logger.info(f"Total dataset size: {size}")
    # logger.info(f"Label distribution in the full dataset: {total_labels}")

    return final_train, final_test
    

def fed_avg(client_params, nodes_list, neigh_list):
    # Initialize the aggregated weights with the current node's parameters
    aggregated_weights = {k: torch.zeros_like(v) for k, v in client_params.items()}
    
    # Accumulate the weights from the neighbors
    for node_id in neigh_list:
        node_weights = nodes_list[node_id]
        
        for k in aggregated_weights.keys():
            aggregated_weights[k] += node_weights[k]
        '''for k in nei_aggregated_weights.keys():
            nei_aggregated_weights[k] += node_weights[k]'''

    # Average the weights (including the current node's weights)
    num_nodes = len(neigh_list) + 1
    for k in aggregated_weights.keys():
        aggregated_weights[k] += client_params[k]
        avg = aggregated_weights[k]/num_nodes
        aggregated_weights[k] = avg.to(aggregated_weights[k].dtype)

    # Apply the aggregated weights to the model
    # with torch.no_grad():  # Ensure gradients are not tracked for this operation
    #     client.model.load_state_dict(aggregated_weights, strict=False)

    # # self.nei_agg_params = nei_aggregated_weights
    # client.aggregated_params = aggregated_weights
    return aggregated_weights

    