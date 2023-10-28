"""
calculates and plots sillouette score, calinski harabasz score and davies bouldin score for clusters
"""

import importlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, cm
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, \
    calinski_harabasz_score


def plot_silhouette(hist, cluster_labels, cluster_number, method):
    """
    input: hist:: pandas Dataframe
            cluster_labels:: pandas Series
            cluster_number:: int
            method:: method
    output: silhouette_avg:: float

    """
    # SOURCE https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    # Create a subplot with 1 row and 2 columns
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)

    # For the silhouette plot, which ranges from -1 to 1
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*nclusters10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(hist) + (cluster_number + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(hist, cluster_labels)
    print("For n_clusters =", cluster_number, "the average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(hist, cluster_labels)

    y_lower = 10
    for i in range(cluster_number):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / cluster_number)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # nclusters10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(
        "Silhouette analysis for " + method + "-clustering on general data with n_clusters = %d"
        % cluster_number,
        fontsize=14,
        fontweight="bold",
    )

    file_savepath = variables.getSavePath('viz', '100_clustering_eval\\' + method + '\sh_n' + str(
        cluster_number) + '_' + method + '_general.png')
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under " + file_savepath)
    plt.clf()

    return silhouette_avg


def cluster_eval(method):
    """
    input: method:: method (for different Evaluation metrics)
    output: --
    """
    # --- general ---
    print("--------- Starting cluster evaluation for method " + method + " ---------")

    s_averages = []
    db_indices = []
    ch_indices = []

    for cluster_number in clusternumbers:
        cluster_data = pd.read_csv(
            variables.getSavePath('data', '070_clustering\\' + method + '\cluster_n' + str(cluster_number) + '.csv'))
        cluster_labels = cluster_data['cluster']
        hist = cluster_data.drop(columns=['cluster', 'time', 'date'])

        # Silhouette plots
        sh = plot_silhouette(hist, cluster_labels, cluster_number, method)
        s_averages.append(sh)

        # Davies-Bouldin Index
        db = davies_bouldin_score(hist, cluster_labels)
        db_indices.append(db)
        print("For n_clusters =", cluster_number, "the average davies_bouldin_score is :", db)

        # Calinski Harabasz Index
        ch = calinski_harabasz_score(hist, cluster_labels)
        ch_indices.append(ch)
        print("For n_clusters =", cluster_number, "the average calinski_harabasz_score is :", ch)

    # plot for average silhouette scores
    fig2 = plt.figure()
    plt.plot(clusternumbers, s_averages, color='maroon', marker='x')
    plt.xticks(clusternumbers)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average Silhouette Score")
    plt.suptitle("Silhouette Score for general model using " + method)
    plt.title("(higher is better)")

    file_savepath = variables.getSavePath('viz', '100_clustering_eval\\' + method + '\sh_' + method + '_general.png')
    plt.ylim(0.5, 1)
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under " + file_savepath)
    plt.clf()

    # plot for Davies-Bouldin Index
    fig3 = plt.figure()
    plt.plot(clusternumbers, db_indices, color='green', marker='x')
    plt.xticks(clusternumbers)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Davies-Bouldin Index")
    plt.suptitle("Davies-Bouldin Index for general model using " + method)
    plt.title("(lower is better)")

    file_savepath = variables.getSavePath('viz', '100_clustering_eval\\' + method + '\db_' + method + '_general.png')
    plt.ylim(0, 1)
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under", file_savepath)
    plt.clf()

    # plot for Calinski Harabasz Index
    fig4 = plt.figure()
    plt.plot(clusternumbers, ch_indices, color='blue', marker='x')
    plt.xticks(clusternumbers)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Calinski-Harabasz Index")
    plt.suptitle("Calinski-Harabasz Index for general model using " + method)
    plt.title("(higher is better)")

    file_savepath = variables.getSavePath('viz', '100_clustering_eval\\' + method + '\ch_' + method + '_general.png')
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under", file_savepath)
    plt.clf()

    # --- personal ---

    data = pd.DataFrame(columns=['cluster_number', 'db', 'ch', 'sh'])

    for cluster_number in clusternumbers:
        for person in variables.personIDs:
            cluster_data = pd.read_csv(
                variables.getSavePath('data',
                                      '070_clustering\\' + method + '_personal\\nclusters' + str(cluster_number) +
                                      '\clusters_n' + str(cluster_number) + '_p' + str(person) + '.csv'))
            cluster_labels = cluster_data['cluster']
            hist = cluster_data.drop(columns=['cluster', 'time', 'date'])

            sh = silhouette_score(hist, cluster_labels)
            db = davies_bouldin_score(hist, cluster_labels)
            ch = calinski_harabasz_score(hist, cluster_labels)

            # SOURCE https://www.geeksforgeeks.org/how-to-append-a-list-as-a-row-to-a-pandas-dataframe-in-python/
            data.loc[len(data)] = [cluster_number, db, ch, sh]

    boxplt = sns.boxplot(data=data, x="cluster_number", y="sh")
    boxplt.set_title("(higher is better)")
    plt.suptitle("Average Silhouette Scores for personal models using " + method)
    file_savepath = variables.getSavePath('viz',
                                          '100_clustering_eval\\' + method + '\sh_' + method + '_boxplot_personal.png')
    plt.ylim(0.5, 1)
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under", file_savepath)
    plt.clf()

    boxplt2 = sns.boxplot(data=data, x="cluster_number", y="db")
    boxplt2.set_title("(lower is better)")
    plt.suptitle("Davies-Bouldin Scores for personal models using " + method)
    file_savepath = variables.getSavePath('viz',
                                          '100_clustering_eval\\' + method + '\db_' + method + '_boxplot_personal.png')
    plt.ylim(0, 1)
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under", file_savepath)
    plt.clf()

    boxplt3 = sns.boxplot(data=data, x="cluster_number", y="ch")
    boxplt3.set_title("(higher is better)")
    plt.suptitle("Calinski-Harabasz Scores for personal models using " + method)
    file_savepath = variables.getSavePath('viz',
                                          '100_clustering_eval\\' + method + '\ch_' + method + '_boxplot_personal.png')
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under", file_savepath)
    plt.clf()


if __name__ == '__main__':
    sns.set_theme()

    variables = importlib.import_module('40_Realisation.code.variables')
    clusternumbers = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    cluster_eval('kmeans')
    cluster_eval('lda')
