import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
from DFL_util import dataset_model_set, create_graph_object
import os
from matplotlib.font_manager import FontProperties

def extract_data_from_directory(directory_path):
    file_pattern = os.path.join(directory_path, f"Round_9.xlsx")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
      raise FileNotFoundError(f"No files found in the directory {directory_path} matching the pattern 'Round_*.xlsx'")
    
    df = pd.read_excel(files[0]) # here since we only extract from Round_9, so the files is single.
    
    return df
    

def _data_prepare(data, graph):
    data = np.array(data.iloc[0:, 0:].values, dtype=float)
    num_nodes = data.shape[0]
    
    # Check if the data matrix is symmetric
    is_symmetric = np.allclose(data, data.T)
    
    # Prepare features and labels
    X = []
    y = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Skip diagonal (self-connection)
                # If the matrix is symmetric, only process one side of the diagonal
                if is_symmetric and i < j:
                    # Feature vector for node pair (i, j)
                    feature_vector = [data[i, j]]
                    X.append(feature_vector)
                    
                    # Label: 1 if there is an edge between nodes i and j, 0 otherwise
                    if graph.has_edge(i, j):
                        y.append(1)  # Direct link in the graph
                    else:
                        y.append(0)  # No direct link in the graph
    
                # If the matrix is asymmetric, process both sides
                elif not is_symmetric:
                    # Feature vector for node pair (i, j)
                    feature_vector = [data[i, j]]
                    X.append(feature_vector)
    
                    # Label: 1 if there is an edge between nodes i and j, 0 otherwise
                    if graph.has_edge(i, j):
                        y.append(1)  # Direct link in the graph
                    else:
                        y.append(0)  # No direct link in the graph
    
    # Convert X and y to NumPy arrays
    X = np.array(X).flatten()
    y = np.array(y)
    
    return X , y
    
    
def vertical_violin_plot(data, metric, save_dir):
    if metric == "Average_Loss":
        metric = "Relative Loss"
    elif metric == "Cosine_Similarity":
          metric = "Cosine Similarity"
    elif metric == "Average_Entropy":
          metric = "Relative Entropy"
    else:
         metric = "Euclidean Similarity"

    plot_data = {
        'Topologies': ['Ring'] * (len(data['ring'][0]) + len(data['ring'][1])) +
                      ['Star'] * (len(data['star'][0]) + len(data['star'][1])) +
                      ['ER_0.5'] * (len(data['ER_0.5'][0]) + len(data['ER_0.5'][1])),
        
        f'{metric}': np.concatenate([data['ring'][0], data['ring'][1],
                                  data['star'][0], data['star'][1],
                                  data['ER_0.5'][0], data['ER_0.5'][1]]),
        
        'Connection Type': ['Edge'] * len(data['ring'][0]) + ['Non-Edge'] * len(data['ring'][1]) +
                           ['Edge'] * len(data['star'][0]) + ['Non-Edge'] * len(data['star'][1]) +
                           ['Edge'] * len(data['ER_0.5'][0]) + ['Non-Edge'] * len(data['ER_0.5'][1])
    }

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(6, 5))
    sns.violinplot(x='Topologies', y=f'{metric}', hue='Connection Type', data=df, split=True, inner='quartile', palette="Set2")

    plt.legend(title='Connection Type', loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.xlabel('Topologies', fontsize=12)  
    plt.ylabel(f'{metric}', fontsize=12)

    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=1.0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f"{metric}.pdf")

    # Save the plot as a PDF
    plt.savefig(file_path)
    plt.close()
        
        
def main():
    DATASET = ["Mnist"]
    TOPOLOGY = ["star", "ring", "ER_0.5"]
    MODEL = ["mlp", "mobile"]
    MAX_EPOCHS = [10]
    METRICS = ["Average_Loss", "Cosine_Similarity", "Euclidean_Distance", "Average_Entropy"]
    NUM = 10 
    ROUND = 9
    
    IID = 1    
    ALPHA = 0.1    
    SEED = 42
    
    for dataset in DATASET:
        for model_name in MODEL:
            # dataset and model setting
            global_dataset, global_model = dataset_model_set(dataset, model_name)
            
            if global_dataset == None or global_model == None:
                continue
                
            for metric in METRICS:
                case_dict = {}

                for topo in TOPOLOGY:  
                    case_dict[topo]=[]
                    
                    G = create_graph_object(NUM, topo)         
                    
                    for max_epoch in MAX_EPOCHS:
                        file_name = dataset + "_" + model_name + "_" + topo + "_" + str(IID) + "_" + str(ALPHA) + "_" + str(SEED) + "_" + str(max_epoch)

                        FL_data = extract_data_from_directory(f"./saved_results/{metric}/Extra_cases_10/{file_name}/")
                        
                        X, y = _data_prepare(FL_data, G)
                        
                        edge_group = X[y == 1]
                        non_edge_group = X[y == 0]
                        
                        case_dict[topo].append(edge_group)
                        case_dict[topo].append(non_edge_group)
                        
                
                vertical_violin_plot(data=case_dict, metric=metric, save_dir="./saved_results/Attack_Plots/Veritcal_Violin")         
                        
                            
                            
                            
if __name__ == "__main__":
    main()
                           
                            
                            
                            
                            
                            
                            
                            