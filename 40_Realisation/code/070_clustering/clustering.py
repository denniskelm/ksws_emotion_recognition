"""
Cluster data with K-means with different cluster sizes (from 1-10) and LDA
"""
import importlib
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation


if __name__ == '__main__':
    variables = importlib.import_module('40_Realisation.code.variables')

    histograms_general = pd.read_csv(variables.getSavePath("data", '060_generate_histograms\histograms.csv'))

    personal_paths = variables.getSavePath('data', '061_generate_histograms_personal')

    # ----- Kmeans -----
    for cluster_number in variables.numbers_of_clusters:
        print('Calculating KMeans cluster with n = ' + str(cluster_number))

        # -- general --
        hist = histograms_general.drop(columns=['time', 'date'])
        kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(hist)
        histograms_general['cluster'] = kmeans.labels_

        histograms_general.to_csv(
            variables.getSavePath('data', '070_clustering\kmeans\cluster_n' + str(cluster_number) + '.csv'), index=False)

        # -- personal --
        for person in variables.personIDs:
            file = personal_paths + "\histogram_" + str(person) + ".csv"
            print('ID: ' + str(person) + ' --- Path: ' + str(file))
            histograms = pd.read_csv(file)

            hist = histograms.drop(columns=['time', 'date'])
            kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(hist)
            histograms['cluster'] = kmeans.labels_

            histograms.to_csv(variables.getSavePath('data', '070_clustering\kmeans_personal\\nclusters' + str(
                cluster_number) + '\clusters_n' + str(cluster_number) + '_p' + str(person) + '.csv'), index=False)


    # ----- LDA -----


    histograms_general = pd.read_csv(variables.getSavePath("data", '060_generate_histograms\histograms.csv'))
    hist_for_lda_ = histograms_general.drop(columns=['time', 'date'])  # drop columns time and date from the data fram

    for cluster_number in variables.numbers_of_clusters:
        print('Calculating general LDA cluster with n = ' + str(cluster_number))

        # --- general ---
        lda = LatentDirichletAllocation(n_components=cluster_number, random_state=0)
        result = lda.fit_transform(hist_for_lda_)

        result_df = pd.DataFrame(result)

        histograms_general['cluster'] = result_df.idxmax(axis='columns')  # Get the maximum of every row
        histograms_general.to_csv(variables.getSavePath('data', '070_clustering\lda\cluster_n' + str(cluster_number) + '.csv'), index=False)

        # --- personal ---
        for person in variables.personIDs:  # for each person
            print("Calculating personal LDA with n = " + str(cluster_number) + " for person ID :", str(person))
            histograms_for_lda = pd.read_csv(
                personal_paths + "\histogram_" + str(person) + ".csv")  # read the histogram result for this person

            hist_for_lda_ = histograms_for_lda.drop(columns=['time', 'date'])  # remove the time and date column
            lda = LatentDirichletAllocation(n_components=cluster_number, random_state=0)
            result = lda.fit_transform(hist_for_lda_)  # apply the model on the data
            result_df = pd.DataFrame(result)  # but the result of the model on the data fram
            histograms_for_lda['cluster'] = result_df.idxmax(axis='columns')  # Get the maximum of every row

            histograms_for_lda.to_csv(variables.getSavePath('data', '070_clustering\lda_personal\\nclusters' + str(
                cluster_number) + '\clusters_n' + str(cluster_number) + '_p' + str(person) + '.csv'),
                                      index=False)  # save the result of LDA clustring on the mentioned pth
