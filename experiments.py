from generate_results import *
from generate_graph import *
from scores_analysis import *
import pandas as pd
import torch
import numpy as np
import pickle

def get_thresholds(average_scores):
    '''the function receives a dictionary of dictionaries with average scores
    of the similarities between the different classes and returns the thresholds list'''
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
    results = {'metric':[], 'threshold':[], 'seed':[]}
    for i in range(11):
        results[f'val{i+1}'] = []
        results[f'test{i+1}'] = []
    for metric, thresholds in metrics_and_thresholds.items():
        for lower_threshold in thresholds:
            for seed in seeds:
                print(f'metric: {metric}, threshold: {lower_threshold}, seed: {seed}') 
                torch.cuda.empty_cache()
                val_acc_lst, test_acc_lst = get_accuracy\
                    (data_df, metric, seed, lower_threshold, float('inf'), True)
                results['metric'].append(metric)
                results['threshold'].append(lower_threshold)
                results['seed'].append(seed)
                for i, (val_acc, test_acc) in enumerate(zip(val_acc_lst, test_acc_lst)):
                    results[f'val{i+1}'].append(val_acc)
                    results[f'test{i+1}'].append(test_acc)
         
        print('temporarily saving resultd')           
        with open('dict_temp_results.pkl', 'wb') as file:
            pickle.dump(results, file)                
    df_results = pd.DataFrame(results)
    return df_results
                    
def main():
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
