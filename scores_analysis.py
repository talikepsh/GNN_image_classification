from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This file is intended to assist with the analysis of scores using boxplots. 
Additionally, it provides a convenient format for organizing and accessing 
the results (similarity scores).
"""

def main_class_vs_all(class_matrix, class_index: int, subject_mapping: dict):
    """
    Calculates the average similarity scores for a given class (specified by index) 
    against itself and all other classes. 

    For each sample in the class specified by `class_index`, the function computes:
    - The average similarity score within the class (excluding the sample itself).
    - The average similarity scores with samples from other classes.

    Args:
        class_matrix: A 2D array where each row represents a sample and 
                    each column represents the similarity scores with all other samples.
        class_index: The index of the class to compute the "1 vs. all" scores for.
        subject_mapping: A dictionary mapping class names to their indices. 
                        Keys are class names, and values are their respective indices.

    Returns:
        dict: A dictionary with class names as keys and lists of average similarity scores as values.
              For the specified class, the list contains scores of each sample against the rest of 
              the class and against samples from other classes.
    """
    classes_names = list(subject_mapping.keys())
    class_dict = dict()
    for name in classes_names:
        class_dict[name] = []
    for i in range(class_matrix.shape[0]):
        relevant_index = list(range(class_index * 500, (class_index+1) * 500))
        same_class_vectors = class_matrix[i][relevant_index]
        class_dict[classes_names[class_index]].append((sum(same_class_vectors) - class_matrix[i][500 * class_index + i]) / (len(same_class_vectors) - 1))
        for j in range(1, 4):
            relevant_index = list(range((class_index - j) * 500, (class_index - j +1) * 500))
            other_class_vectors = [class_matrix[i][k] for k in relevant_index]
            class_dict[classes_names[class_index - j]].append(sum(other_class_vectors) / len(other_class_vectors))
    
    return class_dict

def scores_dicts(similarities, subject_mapping: dict):
    """
    This function calculates the similarity scores of each sample in a given class 
    against itself and other classes (one-vs-rest) and then computes the average 
    similarity scores between each pair of classes.

    Args:
        similarities: A 2D array representing the similarity scores 
                    between samples. The shape of this array is expected 
                    to be (num_samples, num_samples), where each row and 
                    column represents a sample, and the values represent 
                    similarity scores between the samples.
        subject_mapping: A dictionary mapping class names to their respective indices. 
                                Keys are class names, and values are their indices.

    Returns:
        scoring_dict: A nested dictionary where the outer keys are class names and 
                            the inner keys are the class names compared against. The values 
                            are lists of similarity scores of each sample from the main class 
                            to the secondary class.
        averages_dict: A nested dictionary where the outer keys are class names and 
                            the inner keys are class names compared against. The values are 
                            the average similarity scores for all samples in the main class 
                            compared to all samples in the secondary class.
    """
    scoring_dict = dict()
    averages_dict = dict()
    for i, class_name in tqdm(enumerate(subject_mapping)):
        main_class_vectors = similarities[i * 500 : (i + 1) * 500]
        scoring_dict[class_name] = main_class_vs_all(main_class_vectors, i, subject_mapping)
    for main_class in subject_mapping.keys():
        averages_dict[main_class] = dict()
        for secondary_class in subject_mapping.keys():
            scores_list = scoring_dict[main_class][secondary_class]
            averages_dict[main_class][secondary_class] = sum(scores_list) / len(scores_list)
    
    return scoring_dict, averages_dict

def plotting_boxplot(scores: dict):
    """
    This function takes a dictionary containing similarity scores between different
    classes and creates a 2x2 grid of boxplots. Each subplot in the grid represents 
    the distribution of similarity scores for one main class against all other classes, 
    with the box color indicating the main class.

    Args:
        scores: A nested dictionary where the outer keys are class names and the inner keys are 
                class names compared against. The values are lists of similarity scores of each 
                sample from the main class to the secondary class.

    This function displays the boxplots using Matplotlib and does not return any value.
"""
    fig, axis = plt.subplots(2, 2, figsize=(11, 8))
    classes_names = list(scores.keys())
    n_classes = len(classes_names)
    class_list = []
    labels = []
    for main_class in scores.keys():
        for secondary_class in scores[main_class]:
            class_list.append(scores[main_class][secondary_class])
            labels.append(f'{main_class}-{secondary_class}')

    class_index = 0
    for i in range(axis.shape[0]):
        for j in range(axis.shape[1]):
            bplot = axis[i,j].boxplot(class_list[class_index * n_classes : (class_index + 1) * n_classes], positions=range(1, n_classes + 1), patch_artist=True, labels=classes_names)
            colors = ['black' if i==class_index else 'w' for i in range(n_classes)]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            axis[i,j].set_title(classes_names[class_index])
            class_index += 1


    plt.show()