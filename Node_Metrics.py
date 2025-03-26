import numpy as np
import torch
from torch import nn
from collections import defaultdict
from torch.utils.data import Subset, DataLoader
from typing import OrderedDict, List, Optional
import torch.nn.functional as F
import copy
import pandas as pd
import os
import sys


class Node_Metrics:
    def __init__(self, model, w_locals, data_loaders, file_name, dir_name, rou, metric):
        self.model = model # which kind of model used in this DFL
        self.w_locals = w_locals # A list containing all nodes's this round's trained model parameters
        
        self.data_loaders = data_loaders # A list containing all nodes's dataloaders 
        
        self.labels =  [f"node{i}" for i in range(len(w_locals))] # labels depicted in the excel file
        self.file_name = file_name # excel file saved location
        self.dir_name = dir_name # which DFL case
        
        self.rou = rou # which round DFL results
        
        self.metric = metric # which metric we are trying to calcualte this time
        
        
    def execute_attack(self):
        for attr_name in dir(self):
            if attr_name.startswith(self.metric) and callable(getattr(self, attr_name)):
                method = getattr(self, attr_name)
                method()
                
    
    def _save_excel(self, name, df):
        base_directory = f'{self.dir_name}/{name}/{self.file_name}'
        os.makedirs(base_directory, exist_ok=True)
        
        excel_file_path = os.path.join(base_directory, f'Round_{self.rou}.xlsx')
        df.to_excel(excel_file_path, index=False)
        
    
    def loss_attack(self):
        num_node = len(self.data_loaders)  # Number of nodes
        device = next(self.model.parameters()).device
        print(device)
        model = copy.deepcopy(self.model)
        
        # Initialize an empty matrix to store results
        loss_avg = np.zeros((num_node, num_node))
        # loss_sum = np.zeros((num_node, num_node))
        
        model.eval()
    
        for i in range(num_node):  # Iterate over each node's dataloaders
            for j in range(num_node):  # Iterate over each node's models
                local = self.w_locals[j]
                model.load_state_dict(local)  # Load the j-th model's parameters
                
                total_loss = 0.0
                total_samples = 0
    
                with torch.no_grad():
                    for inputs, labels in self.data_loaders[i]:
                        inputs, labels = inputs.to(device), labels.to(device)
    
                        logits = model(inputs)
                        loss = nn.CrossEntropyLoss(reduction='sum')
                        y_loss = loss(logits, labels)
                        total_loss += y_loss.item()
                        total_samples += inputs.size(0)
    
                average_loss = total_loss / total_samples
                loss_avg[i, j] = average_loss
        
        df_avg = pd.DataFrame(loss_avg, index=self.labels, columns=self.labels)
        
        self._save_excel("Average_Loss", df_avg)
        
        
    def cosine_attack(self):
        num_node = len(self.data_loaders)
        cos_sim = np.zeros((num_node, num_node))
        
        for i in range(num_node):
            for j in range(i, num_node):
                if i != j:
                    similarity = self._cosine_metric(self.w_locals[i], self.w_locals[j], True)
                    
                    cos_sim[i, j] = similarity
                    cos_sim[j, i] = similarity  # Since cosine similarity is symmetric
                    
                else:
                    cos_sim[i, j] = 1.0  # Similarity with itself
                    
        df_cos = pd.DataFrame(cos_sim, index=self.labels, columns=self.labels)
        
        self._save_excel("Cosine_Similarity", df_cos)

    
    def _cosine_metric(self, model1: OrderedDict, model2: OrderedDict, similarity: bool = True) -> Optional[float]:
        if model1 is None or model2 is None:
            logging.info("Cosine similarity cannot be computed due to missing model")
            return None
    
        cos_similarities: List = []
    
        for layer in model1:
            if layer in model2:
                l1 = model1[layer].to('cpu')
                l2 = model2[layer].to('cpu')
                if l1.shape != l2.shape:
                    # Adjust the shape of the smaller layer to match the larger layer
                    min_len = min(l1.shape[0], l2.shape[0])
                    l1, l2 = l1[:min_len], l2[:min_len]
                cos = torch.nn.CosineSimilarity(dim=l1.dim() - 1)
                cos_mean = torch.mean(cos(l1.float(), l2.float())).mean()
                cos_similarities.append(cos_mean)
            else:
                logging.info("Layer {} not found in model 2".format(layer))
    
        if cos_similarities:    
            cos = torch.Tensor(cos_similarities)
            avg_cos = torch.mean(cos)
            # relu_cos = torch.nn.functional.relu(avg_cos)  # relu to avoid negative values
            # return relu_cos.item() if similarity else (1 - relu_cos.item())
            return avg_cos.item() if similarity else (1 - avg_cos.item())
        else:
            return None
            
            
    def entropy_attack(self):
        num_node = len(self.data_loaders)
        entropies = np.zeros((num_node, num_node))
        model = copy.deepcopy(self.model)
        
        model.eval()
        device = next(model.parameters()).device
        print([self.file_name, device])

        for i in range(num_node):  # Iterate over each node's dataloaders
            for j in range(num_node):  # Iterate over each node's models
                local = self.w_locals[j]
                model.load_state_dict(local)  # Load the j-th model's parameters
                
                total_entropy = 0.0
                total_samples = 0
    
                with torch.no_grad():
                    for inputs, _ in self.data_loaders[i]:
                        inputs = inputs.to(device)
    
                        logits = model(inputs)
                        probs = F.softmax(logits, dim=1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                        total_entropy += entropy.item() * inputs.size(0)
                        total_samples += inputs.size(0)
    
                average_entropy = total_entropy / total_samples
                entropies[i, j] = average_entropy
    
        df_entropy = pd.DataFrame(entropies, index=self.labels, columns=self.labels)
        
        self._save_excel("Average_Entropy", df_entropy)
        
        
    def euclidean_attack(self):
        num_node = len(self.data_loaders)
        euclidean_distances = np.zeros((num_node, num_node))
        print(self.file_name)
        
        for i in range(num_node):
            for j in range(num_node):
                if i != j:
                    distance = self._euclidean_metric(self.w_locals[i], self.w_locals[j], False, True)
                    euclidean_distances[i, j] = distance
                    euclidean_distances[j, i] = distance  # Since euclidean similarity is symmetric
                else:
                    euclidean_distances[i, j] = 0.0  # Euclidean with itself is zero
                    
        df_euclidean = pd.DataFrame(euclidean_distances, index=self.labels, columns=self.labels)
        
        self._save_excel("Euclidean_Distance", df_euclidean)
    
    
    def _euclidean_metric(self, model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], 
                          standardized: bool = False, similarity: bool = True) -> Optional[float]:
        if model1 is None or model2 is None:
            return None
    
        distances = []
    
        for layer in model1:
            if layer in model2:
                l1 = model1[layer].flatten().to(torch.float32)
                l2 = model2[layer].flatten().to(torch.float32)
                if standardized:
                    l1 = (l1 - l1.mean()) / l1.std()
                    l2 = (l2 - l2.mean()) / l2.std()
                
                distance = torch.norm(l1 - l2, p=2)
                if similarity:
                    norm_sum = torch.norm(l1, p=2) + torch.norm(l2, p=2)
                    similarity_score = 1 - (distance / norm_sum if norm_sum != 0 else 0)
                    distances.append(similarity_score.item())
                else:
                    distances.append(distance.item())
    
        if distances:
            avg_distance = torch.mean(torch.tensor(distances))
            return avg_distance.item()
        else:
            return None
        
    

        