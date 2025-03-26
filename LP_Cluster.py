from Topo_Algo import Topo_Algo
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import copy


class LP_Cluster_Algo(Topo_Algo):
    def __init__(self, case_dict, graph_object, excel_path, attack_path, round_case=None):
        super().__init__(case_dict, graph_object, excel_path, attack_path, round_case)
        
    
    def execute_attack(self):
        for attr_name in dir(self):
            if attr_name.startswith(self.case_dict["Algo"]) and callable(getattr(self, attr_name)):
                method = getattr(self, attr_name)
                method()
        
    
    def _edge_group_clustering(self, value_ls, clustering_name, flag, multi_dimensional=False):
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
        
        # Generate inferred connection list
        inferred_connection = [1 if label == edge_label else 0 for label in labels] # this is a list containing inferred connection, each index value indicating 
                                                                                    # if the current node connecting to this node
        return inferred_connection
        
    
    def Algo_row(self): # this algorithm is used to make the group clustering row by row from the node attributes matrix
        df = self.extract_data_from_directory()
        
        data_dict = {}
    
        for i, row in df.iterrows():
            data_dict[i] = row.tolist() # {0: [...], 1: [...], 2: [...], ..., 9: [...]} 

        inferred_adj = np.zeros((len(data_dict), len(data_dict)-1), dtype=int)
        
        data_dict_copy = copy.deepcopy(data_dict)
    
        for key in range(len(data_dict_copy)): # Here, we try to remove the local metric to exclude the effect from extreme values
            data_dict_copy[key].pop(key) # Besides, here we are trying to modify one mutable onject in Python, so it is
                                           # necessary to make a deepcopy first in case the original version been modified
                                           
        i = 0

        for key, value in data_dict_copy.items():    
            inferred_connection = self._edge_group_clustering(value, self.case_dict["Clustering"], self.case_dict["Flag"])
            
            inferred_adj[i, :] = inferred_connection
            
            i += 1
        
        actual_adj = self.get_actual_adj(self.graph)
        accuracy, precision, fpr, f1 = self.evaluate_inference(inferred_adj, actual_adj)
        
        self.append_experiment_results(self.attack_path, [self.case_dict["File_name"], self.case_dict["Metric"], self.case_dict["Algo"], self.case_dict["Clustering"], accuracy,                                                 precision, fpr, f1,
                                       '\n'.join(','.join(map(str, row)) for row in actual_adj),
                                        '\n'.join(','.join(map(str, row)) for row in inferred_adj)])
                                        

    def Algo_average(self): # this algorithm is used to make the group clustering from the average value of the m_ij and m_ji 
        df = self.extract_data_from_directory()
        
        data = np.array(df.iloc[0:, 0:].values, dtype=float)
        
        rows, cols = data.shape
        averages = []
        
        for i in range(rows): # since the node metrics used here may not be symmetric which means that m_ij perhaps not equal to m_ji, so we try to take the average of them
           for j in range(i + 1, cols):
               upper_value = data[i, j]
               lower_value = data[j, i]
               avg_value = (upper_value + lower_value) / 2
               averages.append(avg_value)
               
        inferred_connection = self._edge_group_clustering(averages, self.case_dict["Clustering"], self.case_dict["Flag"])
        
        inferred_adj_temp = np.zeros((rows, rows))
        k = 0
        for i in range(rows):
            for j in range(i + 1, rows):
                inferred_adj_temp[i, j] = inferred_connection[k]
                k += 1
    
        # Mirror the upper triangle to the lower triangle
        inferred_adj_temp += inferred_adj_temp.T
        
        inferred_adj = np.zeros((rows, rows-1), dtype=int)
    
        for i in range(rows):
            inferred_adj[i] = np.delete(inferred_adj_temp[i], i)
        
        actual_adj = self.get_actual_adj(self.graph)
               
        accuracy, precision, fpr, f1 = self.evaluate_inference(inferred_adj, actual_adj)
        
        self.append_experiment_results(self.attack_path, [self.case_dict["File_name"], self.case_dict["Metric"], self.case_dict["Algo"], self.case_dict["Clustering"], accuracy,                                                 precision, fpr, f1,
                                       '\n'.join(','.join(map(str, row)) for row in actual_adj),
                                        '\n'.join(','.join(map(str, row)) for row in inferred_adj)])
                                        

    
    def Algo_multi(self): # this algorithm is used to make the group clustering from the pairwise tuple value of the m_ij and m_ji 
        df = self.extract_data_from_directory()
        
        data = np.array(df.iloc[0:, 0:].values, dtype=float)
        
        rows, cols = data.shape
        pairwise_metrics = []  # To store (m_ij, m_ji) tuples for each pair
        
        for i in range(rows):
            for j in range(i + 1, cols):  # Iterate through the upper triangle
                m_ij = data[i, j]  # Metric from node i to node j
                m_ji = data[j, i]  # Metric from node j to node i
                pairwise_metrics.append((m_ij, m_ji))  # Store the pair as a tuple
               
        inferred_connection = self._edge_group_clustering(pairwise_metrics, self.case_dict["Clustering"], self.case_dict["Flag"], True)
        
        inferred_adj_temp = np.zeros((rows, rows))
        k = 0
        for i in range(rows):
            for j in range(i + 1, rows):
                inferred_adj_temp[i, j] = inferred_connection[k]
                k += 1
    
        # Mirror the upper triangle to the lower triangle
        inferred_adj_temp += inferred_adj_temp.T
        
        inferred_adj = np.zeros((rows, rows-1), dtype=int)
    
        for i in range(rows):
            inferred_adj[i] = np.delete(inferred_adj_temp[i], i)
        
        actual_adj = self.get_actual_adj(self.graph)
               
        accuracy, precision, fpr, f1 = self.evaluate_inference(inferred_adj, actual_adj)
        
        self.append_experiment_results(self.attack_path, [self.case_dict["File_name"], self.case_dict["Metric"], self.case_dict["Algo"], self.case_dict["Clustering"], accuracy,                                                 precision, fpr, f1,
                                       '\n'.join(','.join(map(str, row)) for row in actual_adj),
                                        '\n'.join(','.join(map(str, row)) for row in inferred_adj)])
                                        
                                        
                                        
    def Algo_column(self):  # this algorithm is used to make the group clustering column by column from the node attributes matrix
        df = self.extract_data_from_directory()
    
        data_dict = {}
    
        # Iterate over the column names, convert "node0" -> 0, "node1" -> 1, etc.
        for col in df.columns:
            # Strip the 'node' prefix and convert to integer
            col_index = int(col.replace('node', ''))  
            data_dict[col_index] = df[col].tolist()  # {0: [...], 1: [...], 2: [...], ...}
    
        inferred_adj = np.zeros((len(data_dict), len(data_dict) - 1), dtype=int)
    
        data_dict_copy = copy.deepcopy(data_dict)
    
        for key in range(len(data_dict_copy)):
            data_dict_copy[key].pop(key)

    
        i = 0
    
        for key, value in data_dict_copy.items():
            inferred_connection = self._edge_group_clustering(value, self.case_dict["Clustering"], self.case_dict["Flag"])
    
            inferred_adj[i, :] = inferred_connection
    
            i += 1
    
        actual_adj = self.get_actual_adj(self.graph)
        accuracy, precision, fpr, f1 = self.evaluate_inference(inferred_adj, actual_adj)
    
        self.append_experiment_results(
            self.attack_path,
            [
                self.case_dict["File_name"], 
                self.case_dict["Metric"], 
                self.case_dict["Algo"], 
                self.case_dict["Clustering"], 
                accuracy, 
                precision, 
                fpr, 
                f1,
                '\n'.join(','.join(map(str, row)) for row in actual_adj),
                '\n'.join(','.join(map(str, row)) for row in inferred_adj),
            ]
        )

                                    
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        