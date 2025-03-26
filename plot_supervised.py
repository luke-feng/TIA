import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from DFL_util import dataset_model_set
import os
from matplotlib.font_manager import FontProperties


def extract_filtered_column_values(file_name, column_name, filters):
    df = pd.read_excel(file_name)
    
    for filter_column, filter_value in filters.items():
        if filter_column in df.columns:
            df = df[df[filter_column] == filter_value]
        else:
            raise ValueError(f"Column '{filter_column}' not found in the Excel file.")
            
    
    if column_name in df.columns:
        filtered_values = df[column_name].values.tolist()
        if len(filtered_values) == 1:
            return filtered_values[0]
        else:
            return filtered_values
    else:
        raise ValueError(f"Column '{column_name}' not found in the Excel file.")
        
        
def plot_topology_bars_advanced(x1, x2, y, save_dir, file_name):
    topologies = x1
    models = x2 
    
    f1_scores = y
    
    x = np.arange(len(topologies))  
    width = 0.15  
    
    fig, ax = plt.subplots(figsize=(12, 4)

    colors = sns.color_palette("Set1", len(models))

    for i, model in enumerate(models):
        model_f1_scores = [f1_scores[topology][i] for topology in topologies]
        ax.bar(x + i * width, model_f1_scores, width, label=model, color=colors[i])

    ax.set_xlabel('Topologies', fontsize=14)
    ax.set_ylabel('F1 Score', fontsize=14)
    ax.set_xticks(x + width * 1.5) 
    ax.set_xticklabels(topologies, fontsize=14)

    max_f1 = max(max(scores) for scores in f1_scores.values()) 
    ax.set_ylim([0, max_f1 + 0.1])  

    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    
    legend_title_font = FontProperties(size=12)

    ax.legend(title='Attack Models', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True, fancybox=True, title_fontproperties=legend_title_font)

    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=1.0)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f"{file_name}.pdf")
    
    # Save the figure as a PDF file
    plt.savefig(file_path, format='pdf')
    
    
def main():
    DATASET = ["Cifar10no", "FMnist", "Mnist", "Cifar10"]
    TOPOLOGY = ["ER_0.3", "star", "ring", "ER_0.5", "ER_0.7"]
    MODEL = ["mlp", "mobile"]
    MAX_EPOCHS = [3]
    METRICS = ["Average_Loss", "Cosine_Similarity", "Euclidean_Distance"] #,, "Average_Entropy"
    ALGO = ["balanced", "SMOTE", "undersampling"]
    ClUSTERING = ["Logit", "RandomForest", "SVM"]
    NUM = [30]#, 20, 30] 
    ROUND = [9, 39]
    
    IID = [1]    
    ALPHA = 0.1    
    SEED = 42
    
    for num in NUM:
        for iid in IID:
            for rou in ROUND:
                for max_epoch in MAX_EPOCHS:
                    for model_name in MODEL:
                            
                        for metric in METRICS: 
                            for algo in ALGO:
                                for dataset in DATASET:
                                    # dataset and model setting
                                    global_dataset, global_model = dataset_model_set(dataset, model_name)
                                    
                                    if global_dataset == None or global_model == None:
                                        continue
                                        
                                    f1_scores = {}
                                    
                                    for topo in TOPOLOGY:
                                        f1_scores[topo] = []
                                        
                                        
                                        file_name = dataset + "_" + model_name + "_" + topo + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(SEED) + "_" + str(max_epoch) + "_" + str                                                        (num)
                                        
                                        
                                        for clustering in ClUSTERING:
                                            file_path = f"./saved_results/Attack_Performance/Extra_cases_{num}/Link_Prediction_Supervised_{iid}_{rou}.xlsx"
                                            
                                            column_to_extract = 'Weighted_F1'
                                            
                                            filters = {
                                                      "File Name": f"{file_name}",  # Filter by a specific file name
                                                      "Metric": f"{metric}",
                                                      "Algo": f"{algo}",
                                                      "Model": f"{clustering}",
                                                      "Ratio": 0.3
                                                      }
                                                      
                                            filtered_column_values = extract_filtered_column_values(file_path, column_to_extract, filters)
                                            
                                            f1_scores[topo].append(filtered_column_values)
                                            
                                    print(file_name)
                                    print(f1_scores)
                                    plot_name = dataset + "_" + model_name + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(SEED) + "_" + str(max_epoch) + "_" + str(num) + "_" + str(rou)
                                    
                                    #print(plot_name)
                                    
                                    plot_topology_bars_advanced(TOPOLOGY, ClUSTERING, f1_scores, f"./saved_results/Attack_Plots/Extra_cases_{num}/{metric}/Supervised/{algo}",                                                                             plot_name)
                                                  
                                    
                                
                            
                            
                            

if __name__ == "__main__":
    main()

