from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import pandas as pd
from skimage.filters import threshold_otsu


def _edge_group_clustering(value_ls, clustering_name, flag, multi_dimensional=False):
        if multi_dimensional:
            # Multi-dimensional array for Algo_multi
            node_metrics_array = np.array(value_ls)
        else:
            # Single-dimensional array for Algo_row, Algo_average, Algo_column
            node_metrics_array = np.array(value_ls).reshape(-1, 1)

        if clustering_name == "Kmeans":
            clustering_model = KMeans(n_clusters=2, init='k-means++', n_init=50, random_state=0)
        elif clustering_name == "GMM":
            clustering_model = GaussianMixture(n_components=2, n_init=50, random_state=0)
        elif clustering_name == "Spectral":
            n_neighbors = min(len(node_metrics_array), 10) # Since in some DFL cases with fewer nodes, the data samples do not satisfy the minimum requirements 
                                                           # for spectral clustering, here we apply a minimum setting for n_neighbors.
                                                           
            clustering_model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=n_neighbors,  random_state=0)
        else:
            raise ValueError("Unsupported clustering method. Choose from 'Kmeans', 'GMM', or 'Spectral'.")

        if clustering_name == "GMM":
            clustering_model.fit(node_metrics_array)
            labels = clustering_model.predict(node_metrics_array)
        else:
            labels = clustering_model.fit_predict(node_metrics_array)
    
        # Calculate mean for each cluster
        cluster_0_mean = node_metrics_array[labels == 0].mean(axis=0).mean() if multi_dimensional else node_metrics_array[labels == 0].mean()
        cluster_1_mean = node_metrics_array[labels == 1].mean(axis=0).mean() if multi_dimensional else node_metrics_array[labels == 1].mean()
        # print(labels[0], cluster_0_mean,cluster_1_mean )
    
        # Determine edge and non-edge labels based on the flag
        if flag:
            # If using negative node metric, smaller mean should correspond to the non-edge group
            if cluster_0_mean < cluster_1_mean:
                edge_label = 1
                non_edge_label = 0
            else:
                edge_label = 0
                non_edge_label = 1
        else:
            # If using positive, smaller mean should correspond to the edge group
            if cluster_0_mean < cluster_1_mean:
                edge_label = 0
                non_edge_label = 1
            else:
                edge_label = 1
                non_edge_label = 0
        
        edge_label = 1 - labels[0]
        non_edge_label = labels[0]
        
        # Generate inferred connection list
        inferred_connection = [1 if label == edge_label else 0 for label in labels] # this is a list containing inferred connection, each index value indicating 
                                                                                    # if the current node connecting to this node
        return inferred_connection, labels
    

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, out_channels, heads=1, concat=False)
        self.norm = torch.nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = self.norm(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, Z):
        N = Z.shape[0]
        pairs = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N)), dim=-1).reshape(-1, 2)
        zi = Z[pairs[:, 0]]
        zj = Z[pairs[:, 1]]
        z_concat = torch.cat([zi, zj], dim=1)
        pred = self.mlp(z_concat).view(N, N)
        return pred

class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        # adj_pred = torch.sigmoid(torch.matmul(z, z.T)) 
        adj_pred = self.decoder(z)
        return adj_pred
    
def build_sparse_edge(sim_matrix, k=3):

    N = sim_matrix.shape[0]
    edge_index = []
    edge_weight = []
    for i in range(N):
        topk = torch.topk(sim_matrix[i], k + 1).indices  # +1 是包含自己
        for j in topk:
            if i != j:
                edge_index.append([i, j])
                edge_weight.append(sim_matrix[i][j])  # 记录边权重
    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # shape [2, E]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)  # shape [E]
    return edge_index, edge_weight


def threshold_by_otsu(adj_pred):
    if hasattr(adj_pred, 'detach'):
        adj_np = adj_pred.detach().cpu().numpy().flatten()
    else:
        adj_np = adj_pred.flatten()
    threshold = threshold_otsu(adj_np)

    edge_matrix = (adj_pred >= threshold).float()
    return edge_matrix, threshold
   
def contrastive_normalization_np_withpower(sim, eta=7):
    sim = np.copy(sim)
    col_min = sim.min(axis=0, keepdims=True)
    col_max = sim.max(axis=0, keepdims=True)
    # Sort each column and get the second largest value
    sorted_cols = np.sort(sim, axis=0)
    second_max = sorted_cols[-2, :].reshape(1, -1)
    
    # col_max = sim.max(axis=0, keepdims=True)

    eps = 1e-8
    normalized = (sim - col_min) / (second_max - col_min + eps)

    normalized = np.clip(normalized, 0, 1)
    normalized = (normalized + normalized.T) / 2
    normalized = np.power(normalized, eta)
    
    return normalized

# Redefine transformation functions using NumPy
def amplify_similarity_np(sim, gamma=5):
    return (1.0 - sim) ** gamma 

def build_sparse_edge_otsu(sim_matrix):
    N = sim_matrix.shape[0]

    sim_np = sim_matrix.cpu().numpy()
    threshold = threshold_otsu(sim_np.flatten())

    edge_index = []
    edge_weight = []

    for i in range(N):
        for j in range(N):
            if i != j and sim_np[i, j] <= threshold:
            # if sim_np[i, j] >= threshold:
                edge_index.append([i, j])
                edge_weight.append(sim_np[i, j].item())

    edge_index = torch.tensor(edge_index, dtype=torch.long).T  # shape [2, E]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)  # shape [E]

    return edge_index, edge_weight, threshold    

def masked_bce_loss(pred, target):
    mask = ~torch.eye(pred.size(0), dtype=torch.bool, device=pred.device)
    loss = F.binary_cross_entropy(pred[mask], target[mask])
    return loss

