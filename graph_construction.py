import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl

"""
This file is designed to efficiently construct a graph in terms of runtime performance.
"""

def construct_nodes(data: pd.DataFrame):
    """
    This function processes the 'embedding' column in the input DataFrame, which contains 
    string representations of feature vectors. It converts these string representations 
    into a list of float numbers, constructs a 2D PyTorch tensor.

    Args:
        data: A DataFrame containing an 'embedding' column where each entry is a string 
            representation of a feature vector.

    Returns:
        torch.Tensor: A 2D tensor of shape (num_samples, num_features) containing the feature vectors 
                      as float values.
    """
    raw_features = data['embedding'].apply(lambda x: (' '.join(x.strip('[').strip(']').strip().split())).split(' '))
    return torch.from_numpy(np.array([[float(val) for val in feature_vector] for feature_vector in raw_features])).to(torch.float32)

def construct_edges(x: torch.tensor ,criteria: list, is_initialized: bool):
    """
    This function computes a similarity matrix for the given node features using the specified metric. 
    It then creates edges between nodes whose similarity falls within a specified range, 
    excluding self-loops.

    Args:
        x: A 2D tensor of shape (num_nodes, num_features) representing the node features.
        criteria: A list containing the metric name (str), lower threshold (float), 
                and upper threshold (float) for edge construction.
        is_initialized (bool): A flag indicating whether to load a precomputed similarity matrix 
                              from disk.

    Returns:
            A 2D tensor of shape (2, num_edges) where the first row contains source nodes 
            and the second row contains target nodes.
    """
    similarities_matrix = get_similarities(x, criteria[0], is_initialized)
    lower_threshold, upper_threshold = criteria[1], criteria[2]
    source, target = [], []
    for i in tqdm(range(similarities_matrix.shape[0])):
        for j in range(similarities_matrix.shape[1]):
            if lower_threshold < similarities_matrix[i][j] < upper_threshold and i != j:
                source.append(i)
                target.append(j)
    return torch.tensor([source, target])

def get_labels(data: pd.DataFrame):
    """
    This function maps each unique class name in the 'class_name' column of the input 
    DataFrame to a unique integer and constructs a tensor of these integer labels.

    Args:
        data: A DataFrame containing a 'class_name' column with categorical labels.

    Returns:
            A 1D tensor containing integer labels for each sample in the input DataFrame.
    """
    subject_mapping = get_subject_mapping(data)
    labels = data['class_name'].apply(lambda x: subject_mapping[x])
    return torch.tensor(labels)

def get_subject_mapping(data: pd.DataFrame):
    """
    This function extracts unique class names from the 'class_name' column of the input 
    DataFrame and assigns each class a unique integer index.

    Args:
        data: A DataFrame containing a 'class_name' column with categorical labels.

    Returns:
            A dictionary mapping each unique class name to a unique integer.
    """
    classes = data['class_name'].unique()
    subject_mapping = dict()
    for i, class_name in enumerate(classes):
        subject_mapping[class_name] = i
    return subject_mapping

def get_similarities(x: torch.tensor, metric: str, is_initialized=True):
    """
    This function supports multiple metrics for similarity/distance computation, including 
    Euclidean distance, city block (Manhattan) distance, max norm, cosine similarity, 
    and chord distance. Optionally, it can load a precomputed similarity matrix from a file.

    Args:
        x: A 2D tensor of shape (num_nodes, num_features) representing the node features.
        metric: The metric to use for similarity/distance computation.
        is_initialized: A flag indicating whether to load a precomputed similarity matrix 
                        from disk.

    Returns:
            A 2D tensor or array containing the similarity/distance values between nodes.
    """
    if is_initialized:
        with open('metrics_dict.pkl', 'rb') as f:
            return pkl.load(f)[metric]
    else:
        if metric == 'euclidian':
            return torch.cdist(x, x, p=2)
        if metric == 'city_block':
            return torch.cdist(x, x, p=1)
        if metric == 'max_norm':
            return torch.max(torch.abs(x[:, None] - x), dim=2).values
        if metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(x)
        if metric ==  'chord':
            norms = torch.norm(x, dim=1, keepdim=True)
            unit_vectors = x / norms
            diff = unit_vectors[:, None, :] - unit_vectors[None, :, :]
            chord_distances = torch.norm(diff, dim=2)
            return chord_distances


def construct_graph(data: pd.DataFrame, criteria: list, is_initialized=True):
    """
    This function constructs node features from embeddings, generates class labels, and 
    constructs edges  based on a specified similarity metric and threshold criteria.
    The resulting graph is represented as node features, labels, and edges.

    Args:
        data: A DataFrame containing 'embedding' and 'class_name' columns.
        criteria: A list containing the metric name (str), lower threshold (float), 
                 and upper threshold (float) for edge construction.
        is_initialized: A flag indicating whether to load a precomputed similarity matrix from disk. 

    Returns:
            x: A 2D tensor of node features.
            y: A 1D tensor of class labels.
            edges: A 2D tensor of edge indices.
    """
    print('Constructing Nodes...')
    x = construct_nodes(data)
    print('Done.')
    print('Retrieving Labels...')
    y = get_labels(data)
    print('Done.')
    print(f'Constructing edges based on: {criteria[0]}')
    edges = construct_edges(x, criteria, is_initialized)
    print('Done, Graph is done constructing.')
    return x, y, edges