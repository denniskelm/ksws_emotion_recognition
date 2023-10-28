"""
Cleans the mood data by only using 18 persons
just little adjustments from original code from Sultana et al.
"""

# Imports
import pandas as pd
import glob
import os
import numpy as np
import importlib

if __name__ == '__main__':
    # import variables module
    variables = importlib.import_module('40_Realisation.code.variables')
    if os.path.isfile(variables.getSavePath("data", "030_clean_mood_data\Mood_primary_18.csv")):
        variables.merged_frame = pd.read_csv(
            variables.getSavePath("data", "030_clean_mood_data\Mood_primary_18.csv"))  # read mode data
        print("Loaded Mood_primary_18.csv in clean_mood_labels.py:")
        print(variables.merged_frame)
    else:
        # Generate new merged_frame

        featuresPath = variables.getSavePath("data",
                                             "020_feature_data\ExtraSensory.per_uuid_features_labels")  # get the path of features
        moodsPath = variables.getSavePath("data",
                                          "020_feature_data\ExtraSensory.per_uuid_mood_labels")  # get the moods path

        # Source Sultana
        features_files = glob.glob(
            featuresPath + "\*.csv.gz")  # store all files names that ends with suffix ".csv.gz" in the folder in a list
        features = [pd.read_csv(file) for file in features_files]  # read all files inside features folder

        mood_files = glob.glob(
            moodsPath + "\*.csv.gz")  # store all files names that ends with suffix ".csv.gz" insode the folder in a list
        moods = [pd.read_csv(file) for file in mood_files]  # read all files inside moods folder



        # Source Sultana

        person = 1
        pList = []  # store person
        list_ = []

        for mood_feature, prim_feature in zip(moods, features):
            mood_data = mood_feature.dropna()  # remove the nuull values from the data
            count_row = mood_data.shape[0]  # get the row
            mood_data['person'] = np.nan  # create new column with null value

            # Select persons having at least 1000 samples
            if mood_data.shape[0] >= 1000:
                pList.append(person)
                mood_data.loc[:, 'person'] = pd.Series(np.repeat(person, count_row), index=mood_data.index)
                # Merge data using inner join of timestamps
                merged_data = pd.merge(mood_data, prim_feature, on='timestamp', how='inner')

                list_.append(merged_data)
                print("Person nr " + str(person) + " has more than 1000 Samples")

            person = person + 1

        variables.merged_frame = pd.concat(list_, axis=0, ignore_index=True, sort=False)

        # Save the merged file

        merged_frame_path = variables.getSavePath("data",
                                                  "030_clean_mood_data\Mood_primary_18.csv")  # get the link where we want to save our result
        variables.merged_frame.to_csv(merged_frame_path)  # save the result
