# GNN_image_classification
This repository contains code and experiments for applying Graph Neural Networks (GNNs) to image classification tasks. Specifically, we use a dataset of image embeddings and construct a graph where each image is represented as a node. The edges are created based on various similarity metrics between the images, and the GNN is used to classify the images by analyzing the structure of the graph.

# Project Overview
In this project, we aim to enhance image classification by leveraging graph-based learning techniques. Instead of treating each image as an isolated input to a neural network, we model the dataset as a graph where images are nodes and edges represent the similarity between pairs of images. The steps include:

Embedding Construction: Using Convolutional Neural Networks (CNNs) to generate embeddings (feature vectors) for each image.
Graph Construction: Constructing a graph from the embeddings using similarity metrics such as cosine similarity, Euclidean distance, and others.
GNN Model: Applying a GNN to this graph to classify the images.
Comparison: Comparing the GNN results to a traditional fully connected network to assess performance improvements.

# Dataset
We use a subset of the ImageNet dataset, specifically focusing on four classes:

Lion
Golden Retriever
Bear
Tarantula
Each class contains 500 samples, resulting in a total dataset size of 2000 images. The dataset is split into:

Training set: 80% (400 samples per class)
Validation set: 10% (50 samples per class)
Test set: 10% (50 samples per class)

# Repository Contents
## Notebooks
analize_results.ipynb: A Jupyter notebook showing how the results of the experiments were analyzed. It includes visualizations and statistical metrics for interpreting the model's performance.

baseline.ipynb: This notebook contains the implementation of a fully connected neural network that serves as a baseline for comparing the GNN's performance.

create_embeddings.ipynb: Demonstrates how the embeddings for the images were constructed using a CNN (credit for bnsreenu/python_for_microscopists/306 - Content based image retrieval​ via feature extraction/VGG_feature_extractor.py). These embeddings are later used to construct the graph.

running_example.ipynb: A running example of the overall pipeline, from embedding generation to graph construction and model evaluation.

## Python Scripts
experiments.py: This script orchestrates the execution of experiments, including training and evaluation of the GNN model across various metrics, thresholds, and random seeds.

graph_construction.py: A module for constructing the graph from the image embeddings based on the similarity metrics (e.g., cosine similarity, Euclidean distance). The graph is built with edges between nodes (images) that satisfy a given similarity threshold.

scores_analysis.py: Performs analysis of the different similarity scores between images, visualizing them using box plots. This helps in understanding how thresholds for graph construction should be chosen.

train_test_main_run.py: The main script that runs the GNN on the constructed graph, evaluates it using different metrics, seeds, and thresholds, and returns the classification accuracy results.

## Data and Results Files
imagenet_embeddings.csv: The CSV file containing the embeddings (feature vectors) of the images after processing through the CNN. This is the input dataset for the graph construction.

dict_results_city_block_cosine.pkl, dict_results_city_block_cosine_chord.pkl, dict_results_euclidian_max_norm.pkl: Pickle files containing dictionaries of the results from experiments using various combinations of similarity metrics.

metrics_dict.pkl: A pickle file storing precomputed metrics (e.g., similarity/distance scores between images) to avoid recalculating them during each run.

results.csv: The final CSV file containing the results of the GNN experiments, including validation and test accuracy for each combination of metric, threshold, and seed.

## Additional Files
dict_results_city_block_cosine.pkl, dict_results_city_block_cosine_chord.pkl, dict_results_euclidian_max_norm.pkl: Pickle files storing the results of various experimental runs using different metrics.

# Key Methodology
## Embedding Construction
The image embeddings are constructed using a pre-trained CNN (e.g., VGG16) to extract feature vectors. These embeddings are a lower-dimensional representation of the original images, which captures important features for classification.

## Graph Construction
After generating embeddings, we compute pairwise similarity scores between images using various metrics such as cosine similarity, Euclidean distance, and others. A threshold is applied to these similarity scores, and an edge is created between two images if their similarity exceeds the threshold. The resulting graph represents the entire image dataset.

## GNN Model
A Graph Neural Network (GNN) is applied to the constructed graph. Each image (node) updates its representation by aggregating information from its neighboring nodes. This allows the model to capture relational information between images, improving classification performance.

## Evaluation
The GNN's performance is compared against a fully connected network (baseline). The results are evaluated using validation and test accuracy metrics. We run multiple experiments with different thresholds, metrics, and random seeds to ensure robustness.


# Future Improvements
Incorporate additional metrics: Experiment with other similarity metrics to explore their impact on graph construction and model performance.
Optimize GNN architecture: Experiment with more advanced GNN architectures or hyperparameter tuning to improve classification accuracy.
Dataset Expansion: Apply the model to larger and more diverse datasets for better generalization.

# Conclusion
This project demonstrates the power of combining image embeddings with graph-based learning methods. By constructing a graph of images and using GNNs, we capture more structural information about the dataset, leading to improved classification results compared to traditional fully connected networks.
