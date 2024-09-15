from generate_results import *
from generate_graph import *
from scores_analysis import *
import pandas as pd
import torch
import numpy as np
import pickle

def get_thresholds(average_scores):
    """
    This function receives a dictionary where each key corresponds to a class, and its value is a 
    nested dictionary containing average similarity scores between that class and other classes. 
    The function extracts the self-similarity score (score of a class with itself) for each class, 
    and calculates an extended range of thresholds by adding small margins below the minimum and 
    above the maximum self-similarity score.

    Args:
        average_scores: A dictionary where each key is a class name and the corresponding 
                        value is another dictionary containing the average similarity 
                        scores between that class and other classes.

    Returns:
        thresholds: A sorted list of threshold values including margins around the 
                    minimum and maximum self-similarity scores.
    """
    min_val = float('inf')
    max_val = float('-inf')
    thresholds = []
    for class_name, vals in average_scores.items():
        thresholds.append(vals[class_name].item())
    for score in thresholds:
        if score > max_val:
            max_val = score
        if score < min_val:
            min_val = score
    thresholds = sorted(thresholds+[min_val*0.95, max_val*1.05])
    return thresholds


def get_results(data_df, metrics_and_thresholds, seeds, ):
    """
    For each metric in `metrics_and_thresholds`, this function iterates over the threshold values 
    and seeds, computes validation and test accuracy, and stores the results. It also temporarily 
    saves the results in a pickle file for backup purposes.

    Args:
        data_df: The input dataset containing image embeddings and class labels.
        metrics_and_thresholds: A dictionary where the key is a metric ('euclidean', 
                                'cosine') and the value is a list of threshold values.
        seeds: A list of random seeds to ensure reproducibility across multiple runs.

    Returns:
        A DataFrame containing the validation and test accuracy results for each 
        metric, threshold, and seed combination.
    """

    results = {'metric':[], 'threshold':[], 'seed':[]}
    for i in range(11):
        results[f'val{i+1}'] = []
        results[f'test{i+1}'] = []
        
    file_name = 'dict_results'    
    for metric, thresholds in metrics_and_thresholds.items():
        for lower_threshold in thresholds:
            for seed in seeds:
                print(f'metric: {metric}, threshold: {lower_threshold}, seed: {seed}') 
                torch.cuda.empty_cache()
                val_acc_lst, test_acc_lst = get_accuracy\
                    (data_df, metric, seed, lower_threshold, float('inf'), 'GraphSAGE', True)
                results['metric'].append(metric)
                results['threshold'].append(lower_threshold)
                results['seed'].append(seed)
                for i, (val_acc, test_acc) in enumerate(zip(val_acc_lst, test_acc_lst)):
                    results[f'val{i+1}'].append(val_acc)
                    results[f'test{i+1}'].append(test_acc)
         
        file_name += f'_{metric}' 
        print('temporarily saving results')           
        with open(f'{file_name}.pkl', 'wb') as file:
            pickle.dump(results, file)                
    df_results = pd.DataFrame(results)
    return df_results
                    
def main():
    """
    The main function that utilize graph construction, similarity score computation, and 
    threshold-based evaluation of accuracy.

    - Loads the dataset containing image embeddings.
    - Constructs node features from the embeddings.
    - Calculates similarity scores for various metrics.
    - Determines appropriate thresholds for each metric based on self-similarity scores.
    - Evaluates model performance using different thresholds and seeds.
    - Saves the results in a CSV file.
    """
    data_df = pd.read_csv('imagenet_embeddings.csv')
    metrics_and_thresholds = {'euclidian':[], 'max_norm':[], 'city_block':[], 'cosine':[], 'chord':[]}
    seeds = [1,2,3,4,5]

    x = construct_nodes(data_df)
    for metric in metrics_and_thresholds.keys():
        sim = get_similarities(x, metric)
        subject_mapping = get_subject_mapping(data_df)
        scores, average_scores = scores_dicts(sim, subject_mapping)
        thresholds = get_thresholds(average_scores)
        metrics_and_thresholds[metric]=thresholds

    results = get_results(data_df, metrics_and_thresholds, seeds)
    results.to_csv('results.csv', index=False)             
                
if __name__ == '__main__':
    main()
