"""
Creating Dataframe for each person 
"""

import pandas as pd
import importlib


variables = importlib.import_module('40_Realisation.code.variables')

path = variables.getSavePath('data', '040_mood_labels\data_with_mood_labels.csv')
df = pd.read_csv(path)

savepath = variables.getSavePath('data', '041_mood_labels_personal')

person_list = df["person"].unique()  # group by person
# print(len(person_list))
for i in person_list:
    # print(i)
    df1 = df[df["person"] == i]
    link = savepath + "/moods_" + str(i) + ".csv"  # the place where we store the file
    df1.to_csv(link, index=False)  # save file

