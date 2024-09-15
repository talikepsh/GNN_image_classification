from generate_results import *
from generate_graph import *
from scores_analysis import *
import pandas as pd
import torch
import numpy as np
import pickle

def get_results(data_df, metrics_and_thresholds, seeds, models):
    results = {'metric':[], 'threshold':[], 'model':[], 'seed':[]}
    for i in range(11):
        results[f'val{i+1}'] = []
        results[f'test{i+1}'] = []
    
    file_name = 'dict_results_architecture'    
    for metric, thresholds in metrics_and_thresholds.items():
        for lower_threshold in thresholds:
            for model in models:
                for seed in seeds:
                    print(f'metric: {metric}, threshold: {lower_threshold}, model: {model}, seed: {seed}') 

                    torch.cuda.empty_cache()
                    val_acc_lst, test_acc_lst = get_accuracy(
                        data_df, metric, seed, lower_threshold, float('inf'), model, True)
                    results['metric'].append(metric)
                    results['threshold'].append(lower_threshold)
                    results['model'].append(model)
                    results['seed'].append(seed)
                    for i, (val_acc, test_acc) in enumerate(zip(val_acc_lst, test_acc_lst)):
                        results[f'val{i+1}'].append(val_acc)
                        results[f'test{i+1}'].append(test_acc)
                        
        print('temporarily saving results')           
        with open(f'{file_name}.pkl', 'wb') as file:
            pickle.dump(results, file)


def main():
    
    data_df = pd.read_csv('imagenet_embeddings.csv')
    seeds = [1,2,3,4,5]
    metrics_and_thresholds = {'chord':[0.7804, 0.7907], 'max_norm': [0.1688],\
        'city_block':[12.290, 12.471, 13.224]}
    models = ['GCN', 'GIN']

    x = construct_nodes(data_df)
    results = get_results(data_df, metrics_and_thresholds, seeds, models)
    results.to_csv('results.csv', index=False) 
        


if __name__ == '__main__':
    main() 