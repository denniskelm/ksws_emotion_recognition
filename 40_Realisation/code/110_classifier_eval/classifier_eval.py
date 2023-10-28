"""
Evaluates Classifiers (Support Vector Machine, Random Forest, Majority Classifier)
"""

import pandas as pd
import numpy as np
import importlib
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import specificity_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import json
import os
import traceback


def classifier(df_c_list, e_array, number_of_folds, p_id, cluster_size):
    """
    input:  df_c_list:: list of pandas.Dataframe (features sorted by time ascending)
            e_array:: numpy.Array containing numpy.Arrays (for each cluster) (targets sorted by time ascending)
            number_of_folds:: int (stratified k fold Cross Validation)
            p_id:: int (Personen ID)
            cluster_size:: int (cluster size)

    Output: metrics_dic:: Dictionary of Dictionaries (Contains metric scores)

    """

    metrics = ['Accuracy_', 'Precision_', 'Recall_', 'Specificity_', 'F1_']
    metrics_dic = {}
    metrics_dummy_dic = {}

    # for cross-validation
    Strat_Kfold = StratifiedKFold(n_splits=number_of_folds)

    # keeping track of scores
    for model in models:
        for metric in metrics:
            metrics_dummy_dic['scores_' + str(metric) + str(model)] = []
            metrics_dic['list_scores_' + str(metric) + str(model) + '_person_' + str(p_id) + '_cluster_size_' + str(
                cluster_size)] = []

    # X = features
    # y = targets

    for cluster, (features, targets) in enumerate(zip(df_c_list, e_array)):
        if cluster_size > 1:
            X = features
            y = targets
        else:
            X = pd.DataFrame(features, columns=[sensor_motion_data])

            y = targets

        for model in models:
            for metric in metrics:
                metrics_dummy_dic['scores_' + str(metric) + str(model)] = []

        try:
            for split_index, (train_index, test_index) in enumerate(Strat_Kfold.split(X, y)):
                # split data
                print("Split: ", str(split_index))

                xtrain_list = []
                ytrain_list = []
                xtest_list = []
                ytest_list = []

                for index in train_index:
                    xtrain_list.append(X.iloc[index].tolist())
                    ytrain_list.append(y[index])

                for index in test_index:
                    xtest_list.append(X.iloc[index].tolist())
                    ytest_list.append(y[index])

                X_train = pd.DataFrame(xtrain_list, columns=sensor_motion_data)
                y_train = ytrain_list
                X_test = pd.DataFrame(xtest_list, columns=sensor_motion_data)
                y_test = ytest_list

                for model in models:
                    print('cluster =', cluster, '| split =', str(split_index), '| model =', str(model), '| person =',
                          str(p_id))

                    if len(np.unique(y_train)) > 1:
                        pred_path = path_pred + "pred_" + "_" + str(model) + '_person_' + str(
                            p_id) + '_cluster_size_' + str(cluster_size) + "_cluster_" + str(cluster) + "_split_" + str(
                            split_index) + ".csv"

                        pred_path_proba = path_pred + "pred_" + "_" + str(model) + '_person_' + str(
                            p_id) + '_cluster_size_' + str(cluster_size) + "_cluster_" + str(cluster) + "_split_" + str(
                            split_index) + "_proba.npy"

                        if os.path.exists(pred_path) and os.path.exists(pred_path_proba):
                            print("Prediction exists")
                            y_pred = pd.read_csv(pred_path, index_col=0)
                        else:
                            print("Prediction does not exist, generating... ")
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            pd.DataFrame(y_pred, columns=['predictions']).to_csv(pred_path)
                            y_probab = model.predict_proba(X_test)
                            np.save(pred_path_proba, y_probab)

                        # calcc
                        score_acc_temp = accuracy_score(y_test, y_pred)
                        metrics_dummy_dic['scores_Accuracy_' + str(model)].append(score_acc_temp)

                        # calcc precision
                        score_pre_RF = precision_score(y_test, y_pred, average='macro')
                        metrics_dummy_dic['scores_Precision_' + str(model)].append(score_pre_RF)

                        # calcc specificity
                        score_spe_RF = specificity_score(y_test, y_pred, average='macro')
                        metrics_dummy_dic['scores_Recall_' + str(model)].append(score_spe_RF)

                        # calcc recall
                        score_rec_RF = recall_score(y_test, y_pred, average='macro')
                        metrics_dummy_dic['scores_Specificity_' + str(model)].append(score_rec_RF)

                        # calcc f1
                        score_f1_RF = f1_score(y_test, y_pred, average='macro')
                        metrics_dummy_dic['scores_F1_' + str(model)].append(score_f1_RF)


                    else:
                        print("y_train has not more than 1 unique class, so no prediction is possible")

                for model in models:
                    for metric in metrics:
                        metrics_dic[
                            'list_scores_' + str(metric) + str(model) + "_person_" + str(p_id) + "_cluster_size_" + str(
                                cluster_size)].append(metrics_dummy_dic['scores_' + str(metric) + str(model)])
        except ValueError as error:
            print("Split not possible because there is an empty cluster: " + str(error))
            traceback.print_exc()
            continue
    return metrics_dic


if __name__ == '__main__':
    SVM = SVC(probability=True)
    RF = RandomForestClassifier(max_leaf_nodes=5)
    DC = DummyClassifier(strategy='most_frequent')
    models = [DC, SVM, RF]

    data_personal = {}
    data_general = {}

    variables = importlib.import_module('40_Realisation.code.variables')

    Clusters = variables.numbers_of_clusters

    path = variables.getSavePath('data', '080_preprocess/')
    path_pred = variables.getSavePath('data', '080_preprocess/pred/')

    sensor_motion_data = variables.sensor_motion_data


    # Accuracy,Precision,Recall,Specifity,AUROC,F1

    for cluster_size in Clusters:
        try:
            print('cluster=_' + str(cluster_size))
            df_c_list = np.load(
                path + '/cluster_size_' + str(cluster_size) + '/general/features.npy',
                allow_pickle=True)
            e_array = np.load(
                path + '/cluster_size_' + str(cluster_size) + '/general/targets.npy',
                allow_pickle=True)
            data_general.update(classifier(df_c_list, e_array, 5, 'general', cluster_size))
        except Exception as e:
            print('general_' + str(cluster_size) + '_didnt work' + str(e))
            pass

        for p_id in variables.personIDs:

            try:
                print('id=' + str(p_id) + '_cluster=' + str(cluster_size))
                df_c_list = np.load(
                    path + '/cluster_size_' + str(cluster_size) + '/personal/features_person_' + str(p_id) + '.npy',
                    allow_pickle=True)
                e_array = np.load(
                    path + '/cluster_size_' + str(cluster_size) + '/personal/targets_person_' + str(p_id) + '.npy',
                    allow_pickle=True)

                data_personal.update(classifier(df_c_list, e_array, 5, p_id, cluster_size))

            except Exception as e:
                print('id=' + str(p_id) + '_cluster=' + str(cluster_size) + '_didnt work: ' + str(e))
                traceback.print_exc()
                pass

    # save dicionary

    json_general = json.dumps(data_general)

    with open(variables.getSavePath('data', '111_classifier_eval_visualization\general.json'), 'w') as a:
        a.write(json_general)

    json_personal = json.dumps(data_personal)

    with open(variables.getSavePath('data', '111_classifier_eval_visualization\personal.json'), 'w') as b:
        b.write(json_personal)
