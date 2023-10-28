"""
generate histograms and histogram violin plots.
"""

import importlib
popko = importlib.import_module('40_Realisation.code.popko')
from datetime import datetime
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import warnings
import matplotlib.cbook
import seaborn as sns


def generate_personal_histograms(id):
    """
    Creates histograms for each person
    ---

    input: id:: int (Personen Id)
    Output: --

    """
    # load csv
    dirname = os.path.dirname

    path = os.path.join(dirname(dirname(dirname(__file__))),
                        os.path.join('data', '041_mood_labels_personal/moods_' + str(id) + ".csv"))
    df = pd.read_csv(path)

    print('ID: ' + str(id) + ' --- Path: ' + path)

    # drop everything execept, 'timesstamp' and 'emotion'
    # the values of emotion are: 1 = discordant, 2 = pleased, ...
    df = df[['timestamp', 'emotion']]  # get those two columns from the data fram

    df['emotion'] = df['emotion'].astype('int')  # convert emotion column to int

    df['datetime'] = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in
                      df.timestamp]  # convert datetime column to timestamp
    df = df[['datetime', 'emotion']]

    df = pd.get_dummies(df, columns=['emotion'])  # build dummy columns from the column emotion

    # needs to be transformed for generate_histogram
    df = df.melt(id_vars=['datetime'], ignore_index=False)
    df = df[df.value > 0]

    # with divisions
    df_histogram = popko.generate_histogram(df, 30)

    # Add columns if they do not exist for violin plot with 6 rows
    for emotion_num in range(1, 7):
        col = 'emotion_' + str(emotion_num)
        if col not in df_histogram:
            df_histogram[col] = 0

    df_histogram = df_histogram.rename(
        columns={'emotion_1': 'discordant', 'emotion_2': 'pleased', 'emotion_3': 'dissuade', 'emotion_4': 'aroused',
                 'emotion_5': 'submissive', 'emotion_6': 'dominance'})  # rename columns
    df_histogram.to_csv(os.path.join(dirname(dirname(dirname(__file__))),
                                     os.path.join('data',
                                                  '061_generate_histograms_personal/histogram_' + str(
                                                      id) + '.csv')))  # Save result to the mentioned path

    fig, axs = plt.subplots(nrows=6, figsize=(15, 50))
    classes = list(df_histogram.columns)  # get all columns as list
    classes.remove('time')  # remove time from the datafram

    colors = np.random.choice(list(mcolors.CSS4_COLORS.keys()), len(classes), replace=True)
    colormap = dict(zip(classes, colors))
    file_savepath = variables.getSavePath('viz',
                                          '061_generate_histograms_personal/histogram_violins_' + str(id) + ".png")

    # show our resluts
    def plot_violins(hist, ax):
        """
        input:  hist:: pandas.Dataframe
                ax:: x-axis
        Output: --
        """
        popko.parallel_coordinates_for_multinomial_distribution(
            hist,
            group_column_name='time',
            class_names=list(colormap.keys()),
            # sub_group_column_name='day',
            # row2linestyle = r2ls,
            kw_lines={
                'color': 'gray',
                'alpha': 0.2,
                'lw': 1,
                'zorder': 0},
            # class_colors=list(colormap.values()),
            ax=ax,
            filename=file_savepath)


    df_histogram_ = df_histogram[(df_histogram['time'] >= '00:00:00') & (df_histogram['time'] < '04:00:00')]
    plot_violins(df_histogram_, axs[0])
    df_histogram_ = df_histogram[(df_histogram['time'] >= '04:00:00') & (df_histogram['time'] < '08:00:00')]
    plot_violins(df_histogram_, axs[1])
    df_histogram_ = df_histogram[(df_histogram['time'] >= '08:00:00') & (df_histogram['time'] < '12:00:00')]
    plot_violins(df_histogram_, axs[2])
    df_histogram_ = df_histogram[(df_histogram['time'] >= '12:00:00') & (df_histogram['time'] < '16:00:00')]
    plot_violins(df_histogram_, axs[3])
    df_histogram_ = df_histogram[(df_histogram['time'] >= '16:00:00') & (df_histogram['time'] < '20:00:00')]
    plot_violins(df_histogram_, axs[4])
    df_histogram_ = df_histogram[(df_histogram['time'] >= '20:00:00') & (df_histogram['time'] < '24:00:00')]
    plot_violins(df_histogram_, axs[5])



# Generate vizualize Histogram for emotion by time
variables = importlib.import_module('40_Realisation.code.variables')
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

sns.set_theme()

for id in variables.personIDs:
    generate_personal_histograms(id)







