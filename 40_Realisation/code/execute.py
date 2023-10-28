import importlib

import pandas as pd
import seaborn as sns

pd.options.mode.chained_assignment = None

variables = importlib.import_module('40_Realisation.code.variables')

# IMPORT ALL CODE HERE with folder_name/python_script_name


def printStep(string):
    print()
    print("--------------------------------------------")
    print("Starting with", string)
    print("--------------------------------------------")
    print()


if __name__ == '__main__':
    variables.init()
    sns.set_theme()

    # Copyright
    print("(c) 2023 Group G01: Jonny Schlutter, Dennis Kelm, Ole Adelmann, Gia Huy Hans Tran, Mhd Esmail Kanaan")
    print("for KSWS/Projekt Smart Computing at the University of Rostock")

    printStep("printing external sources")
    print("Used code and methods by")
    print(
        "M. Popko, S. Bader, S. Lüdtke, and T. Kirste: \"Discovering behavioural predispositions in data to improve human activity recognition,\" in Proceedings of the 7th International Workshop on Sensor-based Activity Recognition and Artificial Intelligence, 2023.")

    print("Used the dataset by")
    print(
        "Y. Vaizman, K. Ellis, and G. Lanckriet, \"Recognizing detailed human context in the wild from smartphones and smartwatches,\" IEEE Pervasive Comput., vol. 16, no. 4, pp. 62–74, 2017.")

    print("Used code and methods by")
    print(
        "M. Sultana, M. Al-Jefri, and J. Lee, \"Using machine learning and smartphone and smartwatch data to detect emotional states and transitions: Exploratory study,\" JMIR MHealth UHealth, vol. 8, no. 9, p. e17818, 2020.")

    printStep("the Emotion prediction pipeline")

    printStep("cleaning the mood data")
    cmd = importlib.import_module('030_clean_mood_data.clean_mood_data')

    printStep("generating mood labels")
    ml = importlib.import_module('040_mood_labels.mood_labels')

    printStep("generating mood visualizations - Part 1")
    mv = importlib.import_module('050_mood_labels_viz.mood_viz')

    printStep("generating mood visualizations - Part 2")
    cmv = importlib.import_module('050_mood_labels_viz.cooler_mood_viz')

    printStep("splitting mood labels in personalized ones")
    dwml = importlib.import_module('060_generate_histograms.data_with_mood_labels')

    printStep("generating histograms and calculating its visualizations")
    gh = importlib.import_module('060_generate_histograms.generate_histograms')

    printStep("clustering")
    cl = importlib.import_module('070_clustering.clustering')

    printStep("clustering visualization")
    clviz = importlib.import_module('071_clustering_viz.clustering_viz')

    """
    Before executing this, install kaleido (can have problems on Windows machines)
    printStep("clustering visualization")
    clviz_ternary = importlib.import_module('070_clustering_viz.cluster_viz_ternary_plot')
    """

    printStep("preprocessing")
    prepro = importlib.import_module('080_Preprocessing.Preprocessing')

    printStep("clustering evaluation")
    cleval = importlib.import_module('100_clustering_eval.clustering_eval')

    printStep("classifying and its evaluation")
    cfy = importlib.import_module('110_classifier_eval.classifier_eval')

    printStep("visualizing the classification evaluation")
    print("Part 1 --- Extracting data for visualization")
    cfy_viz1 = importlib.import_module('111_classifier_eval_viz_data_extraction.viz_data_preprocessing')

    print("Part 2 --- Generating the plots of the evaluation")
    cfy_viz2 = importlib.import_module('112_classifier_eval_viz.classifier_eval_viz')
