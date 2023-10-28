"""
Generates violin plot for each cluster size 
"""

import glob
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# generate plot for clustering
def vplot(file, savepath):
    """
    Generates violin plot
    ---

    input:  file:: CSV data
    output: --
    """
    clusters = pd.read_csv(file)

    classes = list(clusters.columns)
    classes.remove('time')  # remove time from the list
    classes.remove('date')  # remove data from the list
    classes.remove('cluster')  # remove cluster from the list

    colors = np.random.choice(list(mcolors.CSS4_COLORS.keys()), len(classes), replace=False)  # put colours
    colormap = dict(zip(classes, colors))

    fig, ax = plt.subplots(figsize=(15, 5))
    popko.parallel_coordinates_for_multinomial_distribution(
        clusters,
        group_column_name='cluster',
        class_names=list(colormap.keys()),
        kw_lines={
            'color': 'gray',
            'alpha': 0.2,
            'lw': 1,
            'zorder': 0},
        # class_colors=list(colormap.values()),
        ax=ax,
        filename=savepath)


if __name__ == '__main__':
    # vizualization for the clustring results

    variables = importlib.import_module('40_Realisation.code.variables')
    popko = importlib.import_module('40_Realisation.code.popko')

    # --- Kmeans ---
    # -- general --
    path = variables.getSavePath("data", "070_clustering\kmeans")  # get the path for the kmeans result folder
    csv_files = glob.glob(path + "\*.csv")  # store all names on the past folder in a list

    for index, file in enumerate(csv_files):
        savepath = variables.getSavePath("visualizations", '071_clustering_viz/cluster_violins_kmeans_' + str(
            variables.numbers_of_clusters[index]) + ".png")
        vplot(file, savepath)

    for n in variables.numbers_of_clusters:
        path = variables.getSavePath("data", "070_clustering/kmeans_personal/nclusters" + str(n))
        csv_files = glob.glob(path + "\*.csv")

        for i, file in enumerate(csv_files):
            savepath = variables.getSavePath('visualizations',
                                             '071_clustering_viz/personal/viz_n' + str(n) + "_p" + str(
                                                 variables.personIDs[i]) + ".png")
            vplot(file, savepath)
