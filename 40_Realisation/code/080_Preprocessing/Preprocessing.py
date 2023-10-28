"""
Preprocessing data to gather features and targets for classification.
We used following steps:
- assign Motion Data to Cluster
- impute missing data
- divide data into Features and Targets for Classifier
"""

import importlib
import itertools
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


# data cleaning and preperation for the classification model
def preprocessing_data_for_classifier(cluster_size, df_mood_label, df_cluster):
    """
    input: cluster_size:: int
           df_mood_label:: pandas.Dataframe (features for the classification)
           df_cluster:: pandas.Dataframe  (PAD model per time interval with clusters)

    Output: 2-tupel (df_c_list, e_array)
                df_c_list:: list of pandas.Dataframe (features sorted by time ascending)
                e_array:: numpy.Array containing numpy.Arrays (for each cluster) (targets sorted by time ascending)
    """

    cluster = df_cluster
    df = df_raw = df_mood_label  # store the mood labels data in two parameters

    # --- preprocessing ---
    df_raw['date'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in
                      df_raw.timestamp]  # get the data from the timestamp column
    df_raw['time'] = [datetime.utcfromtimestamp(ts).strftime('%H:%M:%S') for ts in
                      df_raw.timestamp]  # get the time from the timestamp column
    df_raw = df.loc[:,
             df_raw.columns.intersection(sensor_motion_data)]  # get all the rows and the sensor motion columns
    df_raw.replace(np.nan, 0, inplace=True)  # fill the null values with 0
    df_raw = df_raw.sort_values(by=['date', 'time'])  # sort the data by date and time

    cluster = cluster.sort_values(by=['date', 'time'])

    cluster_list = list(
        cluster[['date', 'time']].to_records(index=False))  # get the date and time as a list for the data
    df_raw_list = list(df_raw[['date', 'time']].to_records(index=False))

    # assign the motion data to the certain cluster

    list_cluster = []
    for cluster_index, (date_cluster, time_cluster) in enumerate(cluster_list):
        for df_index, (date_df, time_df) in enumerate(df_raw_list):
            if date_df == date_cluster:
                if (datetime.strptime(time_cluster, '%H:%M:%S')) - timedelta(minutes=30) < (
                        datetime.strptime(time_df, '%H:%M:%S')) < (datetime.strptime(time_cluster, '%H:%M:%S')):
                    element = (df_index, (cluster._get_value(cluster_index, 'cluster')))
                    list_cluster.append(element)
            elif date_df > date_cluster:
                continue

    df_raw = df_raw.drop(['time', 'date', 'emotion'], axis=1)

    # fills in gaps with NaN s.t. we can impute later

    try:
        result = [list_cluster[0]]

        for i in range(1, len(list_cluster)):
            index, value = list_cluster[i]
            prev_index = result[-1][0]
            while index > prev_index + 1:
                result.append((prev_index + 1, np.nan))
                prev_index += 1
            result.append((index, value))

        # fill in last tuple with NaN s.t. len(df_raw_list)==len(result)
        for index in range(len(result), len(df_raw_list)):
            result.append((index, np.nan))

        # KNN imputation
        # Alternitavly we could've maybe ignored the missing data for now and only trained on the usable data.
        # and predict the missing data with the resulting model but that might be too tedious and the resulting imporvements might be neglectable
        result_df = pd.DataFrame(result, columns=['index', 'emotion'])
        # build KNNImputer model for completing missing values using k-Nearest Neighbors
        KNN_imp = KNNImputer(missing_values=np.nan, n_neighbors=1, weights="uniform")
        imp_result = KNN_imp.fit_transform(result_df)  # apply our KNNImputer model on the data
        result_imp_df = pd.DataFrame(imp_result, columns=['index', 'CLUSTER'])  # get hte results as a datafram

        result_imp_df = result_imp_df.astype('int')  # casting the result to type int
        result_imp_df = result_imp_df.drop('index', axis=1)  # drop column index from the dataframe

        result_list = result_imp_df.values.tolist()  # convert to list
        result_int_list = list(itertools.chain(*result_list))

        # divide data into

        cluster_data_dic = {}
        emotion_data_dic = {}
        for i in range(cluster_size):
            cluster_data_dic['data_cluster' + str(i)] = []
            emotion_data_dic['data_emotion' + str(i)] = []

        for index, value in enumerate(result_int_list):
            cluster_data_dic['data_cluster' + str(value)].append(df_raw.iloc[index].tolist())
            emotion_data_dic['data_emotion' + str(value)].append(df['emotion'].iloc[index])

        df_c_list = []

        c_array = list(cluster_data_dic.values())
        e_array = list(emotion_data_dic.values())

        for i in c_array:
            df_c_list.append(pd.DataFrame(i, columns=[sensor_motion_data_without_time_date_and_emotions]))

        print(len(df_c_list))
        print('success')
        return df_c_list, e_array

    except Exception as e:
        print('error')
        print(e)
        pass


if __name__ == '__main__':
    variables = importlib.import_module('40_Realisation.code.variables')
    # get paths for our previous results (Clustering, Mode general labels - Mood labels per person)

    save_path = variables.getSavePath('data', '080_preprocess')
    cluster_path = variables.getSavePath('data', '070_clustering')
    df_path = variables.getSavePath('data', '041_mood_labels_personal')
    df_path_general = variables.getSavePath('data', '040_mood_labels')
    df_mood_label = pd.read_csv(df_path_general + '/data_with_mood_labels.csv')

    sensor_motion_data = variables.sensor_motion_data
    sensor_motion_data.extend(['date', 'time', 'emotion'])
    sensor_motion_data_without_time_date_and_emotions = [e for e in sensor_motion_data if
                                                         e not in ('date', 'time',
                                                                   'emotion')]  # get the sensore motion data without time, data and emotion


    Clusters = variables.numbers_of_clusters

    IDs = variables.personIDs
    # apply the previous function on each person
    for cluster_size in Clusters:
        print('cluster size:', cluster_size)
        df_mood_label = pd.read_csv(df_path_general + '/data_with_mood_labels.csv')
        df_cluster = pd.read_csv(cluster_path + '/kmeans/cluster_n' + str(cluster_size) + '.csv')
        df_c_list, e_array = preprocessing_data_for_classifier(cluster_size, df_mood_label, df_cluster)

        np.save(save_path + '/cluster_size_' + str(cluster_size) + '/general/features.npy', np.array(df_c_list))
        np.save(save_path + '/cluster_size_' + str(cluster_size) + '/general/targets.npy', e_array)

        for person_ID in IDs:
            try:
                df_mood_label = pd.read_csv(df_path + '/moods_' + str(person_ID) + '.csv')
                df_cluster = pd.read_csv(
                    cluster_path + '/kmeans_personal' + '/nclusters' + str(cluster_size) + '/clusters_n' + str(
                        cluster_size) + '_p' + str(person_ID) + '.csv')

                df_c_list, e_array = preprocessing_data_for_classifier(cluster_size, df_mood_label, df_cluster)
                np.save(save_path + '/cluster_size_' + str(cluster_size) + '/personal/features_' + "person_" + str(
                    person_ID) + ".npy",
                        np.array(df_c_list))
                np.save(save_path + '/cluster_size_' + str(cluster_size) + '/personal/targets_' + "person_" + str(
                    person_ID) + ".npy",
                        e_array)

            except:
                pass


