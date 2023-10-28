"""
This module creates a stacked plot which shows the Proportion of the 6 emotional states (from the PAD Model) per person
"""

import importlib
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


variables = importlib.import_module('40_Realisation.code.variables')
sns.set_theme(style="white")
# loads csv

print("Creating plot: proportion of emotional state for each person")
path = variables.getSavePath('data', '040_mood_labels\data_with_mood_labels.csv')  # get the mood labels data path

df = pd.read_csv(path)  # read mood labeled data

df = df[['timestamp', 'emotion', 'person']]  # get those 3 columns from the datafram
df['emotion'] = df['emotion'].astype('int')  # convert emotion column to int type
df['datetime'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in
                  df.timestamp]  # build form for timestamp
df = df[['datetime', 'emotion', 'person']]  # get those 3 columns from the datafram
df = pd.get_dummies(df, columns=['emotion'])  # convert emotion column to be as a dummy variables

# creating individual dataframe for each person
df_person_1 = df_person_2 = df_person_3 = df_person_4 = df_person_5 = df_person_6 = df_person_7 = df_person_8 \
    = df_person_9 = df_person_10 = df_person_11 = df_person_12 = df_person_13 = df_person_14 = df_person_15 \
    = df_person_16 = df_person_17 = df_person_18 = pd.DataFrame()

# partitioned into 18 dataframes
df_person_1 = pd.concat([df_person_1, df[df['person'] == 8]])
df_person_2 = pd.concat([df_person_2, df[df['person'] == 10]])
df_person_3 = pd.concat([df_person_3, df[df['person'] == 12]])
df_person_4 = pd.concat([df_person_4, df[df['person'] == 13]])
df_person_5 = pd.concat([df_person_5, df[df['person'] == 15]])
df_person_6 = pd.concat([df_person_6, df[df['person'] == 20]])
df_person_7 = pd.concat([df_person_7, df[df['person'] == 21]])
df_person_8 = pd.concat([df_person_8, df[df['person'] == 25]])
df_person_9 = pd.concat([df_person_9, df[df['person'] == 27]])
df_person_10 = pd.concat([df_person_10, df[df['person'] == 33]])
df_person_11 = pd.concat([df_person_11, df[df['person'] == 35]])
df_person_12 = pd.concat([df_person_12, df[df['person'] == 40]])
df_person_13 = pd.concat([df_person_13, df[df['person'] == 46]])
df_person_14 = pd.concat([df_person_14, df[df['person'] == 48]])
df_person_15 = pd.concat([df_person_15, df[df['person'] == 49]])
df_person_16 = pd.concat([df_person_16, df[df['person'] == 52]])
df_person_17 = pd.concat([df_person_17, df[df['person'] == 54]])
df_person_18 = pd.concat([df_person_18, df[df['person'] == 55]])

# creating a list of all persons
list_person = [df_person_1, df_person_2, df_person_3, df_person_4, df_person_5, df_person_6, df_person_7, df_person_8,
               df_person_9, df_person_10, df_person_11, df_person_12, df_person_13, df_person_14, df_person_15,
               df_person_16, df_person_17, df_person_18]

sample_amounts = [len(df.index) for df in list_person]

list_discordant = []  # emotion 1
list_pleased = []  # emotion 2
list_dissuade = []  # emotion 3
list_aroused = []  # emotion 4
list_submissive = []  # emotion 5
list_dominant = []  # emotion 6

list_person2 = []
for i in list_person:
    i = i.drop('person', axis=1)  # removve  person column
    i = i.melt(id_vars=['datetime'], ignore_index=False)
    i = i[i.value > 0]
    list_person2.append(i)

for person in list_person2:
    a = b = c = d = e = f = 0
    for index, row in person.iterrows():
        if row['variable'] == 'emotion_1':
            a += 1

    for index, row in person.iterrows():
        if row['variable'] == 'emotion_2':
            b += 1

    for index, row in person.iterrows():
        if row['variable'] == 'emotion_3':
            c += 1

    for index, row in person.iterrows():
        if row['variable'] == 'emotion_4':
            d += 1

    for index, row in person.iterrows():
        if row['variable'] == 'emotion_5':
            e += 1

    for index, row in person.iterrows():
        if row['variable'] == 'emotion_6':
            f += 1

    numbers_of_emotion = [a, b, c, d, e, f]
    sum = np.sum(numbers_of_emotion)
    list_discordant.append(a / sum)
    list_pleased.append(b / sum)
    list_dissuade.append(c / sum)
    list_aroused.append(d / sum)
    list_submissive.append(e / sum)
    list_dominant.append(f / sum)

# list to array
array_discordant = np.asarray(list_discordant)
array_pleased = np.asarray(list_pleased)
array_dissuade = np.asarray(list_dissuade)
array_aroused = np.asarray(list_aroused)
array_submissive = np.asarray(list_submissive)
array_dominant = np.asarray(list_dominant)

labels = [str(i) for i in df["person"].unique()]  # group by person
file_savepath = variables.getSavePath('viz',
                                      '050_mood_labels/proportion_of_emotional_state_for_each_person_with_labels.png')

fig, ax = plt.subplots()

colors = [(179 / 255, 226 / 255, 205 / 255), (253 / 255, 205 / 255, 172 / 255),
          (203 / 255, 213 / 255, 232 / 255), (244 / 255, 202 / 255, 228 / 255),
          (230 / 255, 245 / 255, 201 / 255), (255 / 255, 242 / 255, 174 / 255)]

ax.bar(labels, array_discordant, color=colors[0])
ax.bar(labels, array_pleased, bottom=array_discordant, color=colors[3])
ax.bar(labels, array_dissuade, bottom=array_discordant + array_pleased, color=colors[2])  # listemo2
ax.bar(labels, array_aroused, bottom=array_discordant + array_pleased + array_dissuade, color=colors[4])
ax.bar(labels, array_submissive, bottom=array_discordant + array_pleased + array_dissuade + array_aroused,
       color=colors[1])
p1 = ax.bar(labels, array_dominant,
            bottom=array_discordant + array_pleased + array_dissuade + array_aroused + array_submissive,
            color=colors[5])
ax.bar_label(p1, sample_amounts, rotation=90, padding=10)
ax.set_ylim(top=1.19)
ax.set_xlabel('Person ID')
ax.set_ylabel('Proportion of emotional state (%)')
plt.legend(['Discordant', 'Pleased', 'Dissuade', 'Aroused', 'Submissive', 'Dominant'],
           bbox_to_anchor=(1.05, 1.0), loc='upper left')
ax.set_title('Proportion of emotional state for each person (with #samples)')

plt.savefig(file_savepath, bbox_inches='tight', dpi=200)
print("Plot saved under " + file_savepath)
print("Finished creating plot")
