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
from pyhessian import hessian

def _loss_metric(model, received_model_state_dict, local_dataloader, device ='cuda', max_batches=5):
    model.to(device)
    model.eval()
    model.load_state_dict(received_model_state_dict)  # Load the j-th model's parameters
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():       
        for batch_idx, batch in enumerate(local_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            loss = nn.CrossEntropyLoss(reduction='sum')
            y_loss = loss(logits, labels)
            total_loss += y_loss.item()
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def _entropy_metric(model, received_model_state_dict, local_dataloader, device ='cuda', max_batches=5):
    model.to(device)
    model.eval()
    model.load_state_dict(received_model_state_dict)  # Load the j-th model's parameters
    total_entropy = 0.0
    total_samples = 0
    
    with torch.no_grad():       
        for batch_idx, batch in enumerate(local_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            total_entropy += entropy.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_entropy = total_entropy / total_samples
    return average_entropy


def _cosine_metric(model1: OrderedDict, model2: OrderedDict, similarity: bool = True) -> Optional[float]:
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
            print("Layer {} not found in model 2".format(layer))

    if cos_similarities:    
        cos = torch.Tensor(cos_similarities)
        avg_cos = torch.mean(cos)
        # relu_cos = torch.nn.functional.relu(avg_cos)  # relu to avoid negative values
        # return relu_cos.item() if similarity else (1 - relu_cos.item())
        return avg_cos.item() if similarity else (1 - avg_cos.item())
    else:
        return None

    
def _euclidean_metric(model1: OrderedDict[str, torch.Tensor], model2: OrderedDict[str, torch.Tensor], 
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
    

def _jacobian_metric(model, received_model_state_dict, local_dataloader, device ='cuda', max_batches=5):
    model.to(device)
    model.eval()
    model.load_state_dict(received_model_state_dict)  # Load the j-th model's parameters
    total_norm = 0.0
    total_samples = 0
    num_projections = 1
    
    for batch_idx, batch in enumerate(local_dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, _ = batch
        x = x.to(device).detach().requires_grad_(True)

        # Forward
        outputs = model(x)
        
        # Hutchinson trace estimator
        batch_norm = 0.0
        for _ in range(num_projections):
            v = torch.randn_like(outputs)
            grads = torch.autograd.grad(
                outputs,
                x,
                grad_outputs=v,
                retain_graph=True,
                create_graph=False,
                only_inputs=True
            )[0]  # [B, D]
            batch_norm += (grads.norm(dim=1) ** 2).sum().item()  # sum over batch

        batch_norm /= num_projections
        total_norm += batch_norm
        total_samples += x.shape[0]

    avg_frobenius_norm = total_norm / total_samples
    return avg_frobenius_norm


def prepare_input(dataloader, max_batches=1, device='cuda'):
        """ Collects input and target from few batches into single tensors. """
        inputs, targets = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if i >= max_batches:
                    break
                inputs.append(x.to(device))
                targets.append(y.to(device))
        return torch.cat(inputs), torch.cat(targets)

def compute_diag_norm(model, dataloader, max_batches=1, device='cuda'):
    """ Compute Frobenius norm of Hessian diagonal using PyHessian. """
    inputs, targets = prepare_input(dataloader, max_batches)
    model = model.to(device)
    model.eval()

    def loss_fn(output, target):
        return F.cross_entropy(output, target)
    criterion = torch.nn.CrossEntropyLoss()

    hessian_computer = hessian(model=model,
                                data=(inputs, targets),
                                criterion=criterion,
                                cuda=device.startswith('cuda'))

    hess_trace = hessian_computer.trace()
    trace_value = float(torch.tensor(hess_trace).mean())
    # diag_norm = torch.norm(hess_diag).item()
    # print(trace_value)
    return trace_value

def _hessian_metric(model, received_model_state_dict, local_dataloader, device ='cuda', max_batches=5):
    model = copy.deepcopy(model).to(device)
    model.load_state_dict(received_model_state_dict)
    model.eval()

    diag_norm_cur = compute_diag_norm(model, local_dataloader, max_batches=max_batches)
    return diag_norm_cur




def _curvature_metric(model, state_dict_cur_round, state_dict_last_round, local_dataloader, device='cuda', max_batches=5, approach = 'hess'):
    cur_model = copy.deepcopy(model).to(device)
    last_model = copy.deepcopy(model).to(device)
    cur_model.load_state_dict(state_dict_cur_round)
    last_model.load_state_dict(state_dict_last_round)
    cur_model.eval()
    last_model.eval()

    
    diag_norm_cur = compute_diag_norm(cur_model, local_dataloader, max_batches=max_batches)
    diag_norm_last = compute_diag_norm(last_model, local_dataloader, max_batches=max_batches)

    rel_curvature = abs(diag_norm_cur - diag_norm_last)
    return rel_curvature



    # for batch_idx, (x, _) in enumerate(local_dataloader):
    #     if batch_idx >= max_batches:
    #         break
    #     x = x.to(device)
    #     x.requires_grad_(True)


    #     # with torch.no_grad():
    #     #     out_cur = cur_model(x)
    #     #     out_last = last_model(x)
    #     # diff = (out_cur - out_last).sum(dim=1)


    #     for i in range(x.size(0)):
    #         xi = x[i:i+1].detach().clone().requires_grad_(True)

    #         def scalar_func(input_x):
    #             out_t = cur_model(input_x)
    #             out_t1 = last_model(input_x)
    #             return (out_t - out_t1).sum()

    #         # hess = hessian(scalar_func, xi)[0] 
    #         if approach == 'approx':
    #             print(scalar_func(xi))
    #             grad1 = torch.autograd.grad(scalar_func(xi), xi, create_graph=True)[0]
    #             print(grad1)
    #             hess_diag = []

    #             for idx in range(grad1.numel()):
    #                 grad2 = torch.autograd.grad(grad1.view(-1)[idx], xi, retain_graph=True)[0]
    #                 hess_diag.append(grad2.view(-1)[idx].item()) 

    #             print(hess_diag)
    #             hess_diag_tensor = torch.tensor(hess_diag, device=xi.device)
    #             print(hess_diag_tensor)
    #             curvature_score = torch.norm(hess_diag_tensor, p=2).item() 
    #             print(curvature_score)
    #             break    
    #             curvature_list.append(curvature_score)
    #         elif approach == 'hess':
    #             hess = hessian(scalar_func, xi)[0] 
    #             hess_norm = torch.norm(hess, p='fro').item()
    #             curvature_list.append(hess_norm)
                
            

    # return np.mean(curvature_list)


def _curvature_divergence(model_a_t: OrderedDict[str, torch.Tensor],
                          model_a_t1: OrderedDict[str, torch.Tensor],
                          model_b_t: OrderedDict[str, torch.Tensor],
                          model_b_t1: OrderedDict[str, torch.Tensor],
                          standardized: bool = False) -> Optional[float]:
    """
    Compute curvature similarity between two models' parameter updates across consecutive rounds.
    
    model_a_t / model_a_t1: parameters of model a at round t and t-1
    model_b_t / model_b_t1: parameters of model b at round t and t-1
    """
    if not (model_a_t and model_a_t1 and model_b_t and model_b_t1):
        return None

    similarities = []

    for layer in model_a_t:
        if layer in model_a_t1 and layer in model_b_t and layer in model_b_t1:

            delta_a = (model_a_t[layer] - model_a_t1[layer]).flatten().to(torch.float32)
            delta_b = (model_b_t[layer] - model_b_t1[layer]).flatten().to(torch.float32)

            if standardized:
                delta_a = (delta_a - delta_a.mean()) / (delta_a.std() + 1e-8)
                delta_b = (delta_b - delta_b.mean()) / (delta_b.std() + 1e-8)

            norm_diff = torch.norm(delta_a - delta_b, p=2)
            norm_sum = 0.5 * (torch.norm(delta_a, p=2) + torch.norm(delta_b, p=2))

            if norm_sum != 0:
                similarity = norm_diff / norm_sum
            else:
                similarity = 1.0  # Maximum dissimilarity if both updates are zero

            similarities.append(similarity.item())

    if similarities:
        return float(torch.mean(torch.tensor(similarities)))
    else:
        return None
    
    