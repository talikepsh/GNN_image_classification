from tqdm import tqdm
import matplotlib.pyplot as plt


def main_class_vs_all(class_matrix, class_index: int, subject_mapping: dict):
    classes_names = list(subject_mapping.keys())
    class_dict = dict()
    for name in classes_names:
        class_dict[name] = []
    for i in range(class_matrix.shape[0]):
        relevant_index = list(range(class_index * 500, (class_index+1) * 500))
        same_class_vectors = class_matrix[i][relevant_index]
        class_dict[classes_names[class_index]].append((sum(same_class_vectors) - 1) / (len(same_class_vectors) - 1))
        for j in range(1, 4):
            relevant_index = list(range((class_index - j) * 500, (class_index - j +1) * 500))
            other_class_vectors = [class_matrix[i][k] for k in relevant_index]
            class_dict[classes_names[class_index - j]].append(sum(other_class_vectors) / len(other_class_vectors))
    
    return class_dict

def scores_dicts(similarities, subject_mapping: dict):
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
    fig, axis = plt.subplots(2, 2, figsize=(15, 15))
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