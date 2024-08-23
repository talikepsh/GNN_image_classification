import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl

def construct_nodes(data: pd.DataFrame):
    raw_features = data['embedding'].apply(lambda x: (' '.join(x.strip('[').strip(']').strip().split())).split(' '))
    return torch.from_numpy(np.array([[float(val) for val in feature_vector] for feature_vector in raw_features])).to(torch.float32)

def construct_edges(x: torch.tensor ,criteria: list, is_initialized: bool):
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
    subject_mapping = get_subject_mapping(data)
    labels = data['class_name'].apply(lambda x: subject_mapping[x])
    return torch.tensor(labels)

def get_subject_mapping(data: pd.DataFrame):
    classes = data['class_name'].unique()
    subject_mapping = dict()
    for i, class_name in enumerate(classes):
        subject_mapping[class_name] = i
    return subject_mapping

def get_similarities(x: torch.tensor, metric: str, is_initialized=True):
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