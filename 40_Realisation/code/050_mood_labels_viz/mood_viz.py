"""
This module creates a heatmap which shows the Mood Distribution per person 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


dirname = os.path.dirname  # get the exist path
path = os.path.join(dirname(dirname(dirname(__file__))),
                    os.path.join('data', '040_mood_labels\data_with_mood_labels.csv'))

df = pd.read_csv(path)  # read the bath od mood labels

# how many occurrences exist for each PDA-Mood
# emotion_counts = pd.value_counts(df['emotion'])

# how many entries are associated with each person
label_counts_per_person = pd.value_counts(df['person'])  # count how many labels for each person

# count the occurences of each mood for each person
mcounts = {}
for index, row in df[['person', 'emotion']].iterrows():
    if int(row['person']) not in mcounts:
        mcounts[int(row['person'])] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    mcounts[int(row['person'])][int(row['emotion'])] += 1 / 63647

print(mcounts[8])

# put everything into an array
data = np.empty((6, 18))
for i, person in enumerate(mcounts):
    for j, mood in enumerate(mcounts[person]):
        data[j, i] = mcounts[person][mood]

# --- Plotting ---

fig, ax = plt.subplots()
im = ax.imshow(data)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.55)
cbar.ax.set_ylabel('percentage out of 63647 datapoints', rotation=-90, va="bottom")
cbar.set_ticklabels(['0%', '1%', '2%', '3%', '4%', '5%', '6%', '7%'])

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(mcounts.keys())))
ax.set_xticklabels(str(x) for x in mcounts.keys())
ax.set_yticks(np.arange(len(mcounts[8].keys())))
ax.set_yticklabels(['discordant', 'pleased', 'dissuade', 'aroused', 'submissive', 'dominance'])

ax.set_title("Mood Distribution per Person (ID)")
fig.tight_layout()

file_savepath = os.path.join(dirname(dirname(dirname(__file__))),
                             os.path.join('visualizations',
                                          '050_mood_labels/moods_per_person.png'))  # Path where we want to save our result

plt.savefig(file_savepath, dpi=200)  # save our results
print("Plot saved under " + file_savepath)
