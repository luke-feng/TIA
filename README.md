# **Novel Topology Inference Attacks on DFL**

This repository contains all the relevant information about Yuanzhe Gao's UZH master's thesis, **Novel Topology Inference Attacks on DFL**, including the project code, experimental results, visualized plots, and the final presentation slides.

## Overview

If you would like to practice or review the inference attacks on DFL topology implemented in this project, please follow the process described below.

## Project Structure

This project contains the following main scripts:

1. **`main_fed.py`**: The primary script for initiating decentralized federated learning (DFL) simulations. This script allows users to configure key DFL parameters and run experiments under various settings. Below are the main configurable options and functionalities supported by `main_fed.py`:

   - **Number of Nodes**: Defines the number of participating nodes in the DFL network. Each node trains a local model on its subset of data.
   - **Network Topology**: Supports different network topologies, influencing how nodes communicate and share information. Configurable topologies can represent various realistic network scenarios.
   - **Custom Datasets and Models**: Users can specify which dataset (e.g., CIFAR-10, MNIST) and model architecture (e.g., CNN, ResNet) each node uses. The flexibility allows testing DFL under different data and model conditions.
   - **Data Distribution (IID vs. Non-IID)**: The dataset can be distributed to nodes in either an IID (Independent and Identically Distributed) manner, where each node has a similar data distribution, or a Non-IID setting, where data distributions vary across nodes. This setting impacts model convergence and performance in DFL.
   - **Local Training Epochs**: Controls the number of epochs each node uses to train its model on the local dataset before participating in aggregation. Adjusting this parameter helps simulate different computational capabilities or training preferences among nodes.
   - **Total Rounds**: Defines the overall number of federated rounds, where each round includes model training on nodes and parameter aggregation.

   ### Saved Models for Topology Inference Attacks

   The `saved_models` folder contains pre-trained models from each DFL node, organized for direct evaluation in topology inference attacks. The naming convention for each model file is as follows:

   ```markdown
   file_name = dataset + "_" + model_name + "_" + topology + "_" + str(iid) + "_" + str(ALPHA) + "_" + str(seed) + "_" + str(max_epoch) + "_" + str(num)

2. **`metric_cal.py`**: This script calculates various node metrics between pairs of nodes under different decentralized federated learning (DFL) conditions. These metrics form the foundation for performing topology inference attacks, providing insights into node relationships and interactions within the network. The primary node metrics calculated by this script include:

   - **Relative Loss**: Measures the difference in model performance between nodes by evaluating how well one node’s model performs on another node’s data. This metric helps quantify the generalizability of models across nodes.
   - **Relative Entropy**: Calculates the entropy difference between nodes, reflecting the uncertainty or confidence levels in predictions when one node’s model is applied to another’s data.
   - **Cosine Similarity**: Computes the cosine similarity between the model parameters of two nodes, providing a measure of alignment in model orientation.
   - **Euclidean Similarity**: Measures the Euclidean distance between model parameters across nodes, offering insights into the magnitude of differences between node models.
   
These metrics collectively enable the analysis of node relationships and serve as the basis for launching topology inference attacks on DFL networks. By examining these metrics, researchers can uncover patterns and infer potential network topologies. The node metric values for various DFL scenarios are stored in the `saved_results` folder, using the same naming convention as described in the previous section.

3. **`metric_analyze.py`**: This script is responsible for executing various topology inference attack methods based on calculated node metrics. Depending on the attack approach, it can be categorized into the following main types:

   - **Supervised Learning-Based Attack**: In this approach, supervised machine learning models are used to perform topology inference. The models supported include:
     - **Logistic Regression**
     - **Random Forest**
     - **Support Vector Machine (SVM)**

     To address imbalanced data distribution, this approach incorporates different data processing algorithms, including:
     - **Balanced**: Applies class weighting to handle imbalanced classes.
     - **SMOTE (Oversampling)**: Synthetic Minority Over-sampling Technique to balance classes by generating synthetic samples.
     - **Undersampling**: Reduces the majority class to achieve balanced class distribution.

   - **Clustering-Based Attack**: This approach uses clustering methods from unsupervised machine learning to infer the network topology. The supported clustering models are:
     - **K-Means Clustering**
     - **Gaussian Mixture Model (GMM)**
     - **Spectral Clustering**

The results of these topology inference attacks are stored in the `saved_results/Attack_Performance` directory, and the corresponding visualizations are saved in `saved_results/Attack_Plots`.

