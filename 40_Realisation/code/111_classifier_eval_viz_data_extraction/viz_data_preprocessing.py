"""
Preprocess Data for visualization with following steps:
- get an average of all different cluster per cluster size
- sort the metrics dictionary 
- save dictionary as json 
"""

import importlib
import json
import statistics


def avg_dict(dict):
    """
    input: dict:: Dictionary
    output: dict1:: Dictionary
    """
    if bool(dict):
        dict1 = {}
        cluster = 1
        key = list(dict.keys())[0]
        while cluster < 11:
            a = 0
            k = 0
            try:
                value = list(dict.values())[cluster - 1]
            except:
                dict1[key[:len(key) - 1] + str(cluster)] = -1
                cluster += 1
                continue
            for i in value:
                try:
                    a += statistics.mean(i)
                    k += 1
                except:
                    continue
            a /= int(k)
            dict1[key[:len(key) - 1] + str(cluster)] = a
            cluster += 1
        return dict1
    else:
        return


def prepare_dict(dict):
    """

    input: dict:: Dictionary
    output: dict1:: Dictionary

    """
    dict1 = {}
    for key, value in dict.items():
        list1 = []
        try:
            a = avg_dict(value)
            for key1, value1 in a.items():
                list1.append(value1)
            dict1[key] = list1
        except:
            dict1[key] = []
    return dict1


if __name__ == '__main__':
    variables = importlib.import_module('40_Realisation.code.variables')

    f = open(variables.getSavePath('data', '111_classifier_eval_visualization\personal.json'), 'r')

    g = open(variables.getSavePath('data', '111_classifier_eval_visualization\general.json'), 'r')

    data_personal = json.load(f)
    data_general = json.load(g)

    metrics = ['Accuracy_', 'Precision_', 'Recall_', 'Specificity_', 'F1_']
    models = ['DummyClassifier', 'SVC', 'RandomForest']
    IDs = [8, 10, 12, 13, 15, 20, 21, 25, 27, 33, 35, 40, 46, 48, 49, 52, 54, 55]


    general_data_scores_dict = {}
    for model in models:
        for metric in metrics:
            general_data_scores_dict[metric + model] = {key: value for key, value in data_general.items() if
                                                        metric + model in key}

    personal_data_persons = {}
    for id in IDs:
        personal_data_persons['person_' + str(id)] = {key: value for key, value in data_personal.items() if
                                                      'person_' + str(id) in key}

    personal_data_scores_dict = {}
    for model in models:
        for metric in metrics:
            for key, value in personal_data_persons.items():
                personal_data_scores_dict[metric + model + '_' + key] = {key1: value1 for key1, value1 in value.items() if
                                                                         metric + model in key1}

    general_data_scores = prepare_dict(general_data_scores_dict)
    personal_data_scores = prepare_dict(personal_data_scores_dict)
    general_data = {}
    personal_data = {}
    for model in models:
        general_data[model] = {key: value for key, value in general_data_scores.items() if model in key}
        personal_data[model] = {key: value for key, value in personal_data_scores.items() if model in key}

    json_general = json.dumps(general_data)

    with open(variables.getSavePath('data', '111_classifier_eval_visualization/') + 'general.json', 'w') as gen:
        gen.write(json_general)

    json_personal = json.dumps(personal_data)

    with open(variables.getSavePath('data', '111_classifier_eval_visualization/') + 'personal.json', 'w') as per:
        per.write(json_personal)
