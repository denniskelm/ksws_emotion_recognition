"""
Generates plots to visualize our results. Following metrics has been used: Accuracy, Precision_, Recall, F1 Score, Specificity.
"""

import importlib
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


def get_personal_means(metric, classifier):
    """
    input:  metric:: string
            classifier:: string

    output: means:: List of Floats
    """
    means = []

    for person in variables.personIDs:
        data = data_personal[classifier][metric + "_" + classifier + "_person_" + str(person)]
        data = [x if x != -1 else np.nan for x in data]
        if not data: means.append([0])
        means.append(data)

    non_zeros = np.count_nonzero(means, axis=0)
    means = np.nanmean(means, axis=0)
    return means


def get_personal_std(metric, classifier):
    """
    input:  metric:: string
            classifier:: string

    output: stds:: List of Floats
    """
    stds = []

    for person in variables.personIDs:
        data = data_personal[classifier][metric + "_" + classifier + "_person_" + str(person)]
        data = [x if x != -1 else np.nan for x in data]
        if not data: stds.append([0])
        stds.append(data)

    non_zeros = np.count_nonzero(stds, axis=0)
    stds = np.nanstd(stds, axis=0)
    return stds


def eval_viz(metric, classifier):
    """
      input:  metric:: string
              classifier:: string

      output: --
    """

    # --- multiplot lines ---
    print("Plotting", metric, 'for', classifier)

    fig, axs = plt.subplots(6, 3)
    fig.suptitle(metric + " for each cluster amount by person, when using " + classifier, y=0.95, fontsize=16)
    fig.set_size_inches(8, 10)

    for i, ax in enumerate(axs.flat):
        ax.set_title('P' + str(IDs[i]), x=0.9, y=1.0, pad=-14)

        plt.sca(ax)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.ylim(0, 1)
        plt.xticks(range(10), clusternumbers)
        plt.xlim(0, 9)
        ax.label_outer()

        data = data_personal[classifier][metric + "_" + classifier + "_person_" + str(IDs[i])]
        data = [x if x != -1 else np.nan for x in data]
        ax.plot(data, color='maroon', marker=4, markerfacecolor='black')

    file_savepath = variables.getSavePath('viz',
                                          '112_classifier_eval_viz\\' + classifier + "\\" + metric + "_by_person.png")
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under", file_savepath)
    plt.clf()

    # --- Barplot ---

    # set width of bars
    barWidth = 0.25

    # set heights of bars
    bars1 = [x if x != -1 else 0 for x in data_general[classifier][metric + "_" + classifier]]
    bars2 = get_personal_means(metric, classifier)

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))  #
    r2 = [x + barWidth for x in r1]

    print(data_general)

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='general')
    plt.bar(r2, bars2, color='#557f2d',
            width=barWidth,  # [barWidth * (np.sqrt(x) / 6) + 0.07 for x in [18, 17, 15, 10, 7, 6, 3, 1, 1, 1]],
            label='personal avg.')

    plt.suptitle(metric + ' for each cluster amount, when using ' + classifier)
    plt.title('general vs avg. personal')  # , box width: #persons with data for that cluster amount')
    plt.ylabel(metric, fontweight='bold')
    plt.ylim(0, 1)
    plt.xlabel('number of clusters', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], clusternumbers)

    # Create legend & Show graphic
    plt.legend()
    fig.set_size_inches(8, 5)
    file_savepath = variables.getSavePath('viz',
                                          '112_classifier_eval_viz\\' + classifier + "\\" + metric + "_personal_average_and_general.png")
    plt.savefig(file_savepath, dpi=200)
    print("Plot saved under", file_savepath)
    plt.clf()


def plot_classifier_comparison(metric, classifiers):
    """
    input:  metric:: string
            classifier:: string

    output: --
    """
    print('Plotting comparison on ' + metric)
    plt.suptitle('Comparing Classifiers on ' + metric)
    plt.title('General model and means of personal models with standard deviation')

    plt.ylim(0, 1)

    plt.xlabel('Number of Clusters')
    plt.xticks(range(0, 10), range(1, 11))

    colors = [(.8, 0, 0, 1), 'green', (0, 0, .5, 1)]
    colorslight = [(.8, 0, 0, 0.5), (0, .6, 0, 0.5), (0, 0, .5, 0.5)]
    colorsgeneral = ['red', (0, .7, 0, 1), 'blue']
    general_y = [None, None, None]
    personals_y = [None, None, None]

    for i, classifier in enumerate(classifiers):
        # Plot general comparison
        general_y[i] = [x if x != -1 else 0 for x in data_general[classifier][metric + "_" + classifier]]
        plt.plot((np.arange(10) + ((i - 1) * 0.07)), general_y[i], marker="x",
                 color=colorsgeneral[i], markersize=5)

        # Plot personal comparison with errorbars
        personals_y[i] = get_personal_means(metric, classifier)
        classifier_label = ''
        if classifier == 'SVC':
            classifier_label = 'SVM'
        elif classifier == 'DummyClassifier':
            classifier_label = 'Majority'
        else:
            classifier_label = 'Random Forest'

        plt.errorbar(y=personals_y[i], x=(np.arange(10) + ((i - 1) * 0.07)), label=classifier_label,
                     yerr=get_personal_std(metric, classifier), capsize=3, elinewidth=1.5, ecolor=colorslight[i],
                     marker="o",
                     color=colors[i], linestyle='dashed', markersize=5)

    if metric != 'Recall':
        plt.fill_between(np.arange(10), personals_y[0], personals_y[2], alpha=0.15, color='c')
        plt.fill_between(np.arange(10), general_y[1], general_y[0], alpha=0.15, color='tab:purple')

    plt.margins(0)

    if metric != 'Accuracy':
        plt.gca().add_artist(plt.legend())
    else:
        plt.gca().add_artist(plt.legend(loc='lower left'))

    # Second legend
    line_solid = mlines.Line2D([1], [1], marker="x", color='dimgrey', linestyle='-', linewidth=1.5, label='General')
    line_dashed = mlines.Line2D([1], [1], marker="o", markersize=3.5, color='black', linestyle='--', linewidth=1.5,
                                label='Personal')

    if metric != 'Recall':
        plt.legend(handles=[line_dashed, line_solid], loc='upper left', labelcolor=['c', 'tab:purple'])
    else:
        plt.legend(handles=[line_dashed, line_solid], loc='lower right')

    file_savepath = variables.getSavePath('viz', '112_classifier_eval_viz\comparison\\' + metric + '.png')
    plt.savefig(file_savepath, dpi=300)
    print("Plot saved under", file_savepath)
    plt.clf()

if __name__ == '__main__':
    variables = importlib.import_module('40_Realisation.code.variables')
    popko = importlib.import_module('40_Realisation.code.popko')

    sns.set_theme(style="whitegrid")

    f = open(variables.getSavePath('data', '111_classifier_eval_visualization/') + 'personal.json', 'r')
    g = open(variables.getSavePath('data', '111_classifier_eval_visualization/') + 'general.json', 'r')

    data_personal = json.load(f)
    data_general = json.load(g)

    IDs = variables.personIDs
    clusternumbers = variables.numbers_of_clusters

    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']  # , 'AUROC']
    classifiers = ['DummyClassifier', 'SVC', 'RandomForest']
    # clusterer = [...]

    """
    print('#-- Classifiers --#')
    for classifier in classifiers:
        print(' -- ' + classifier + ' -- ')
        for metric in metrics:
            eval_viz(metric, classifier)
    """
    print('\n', '#-- Comparison --#')
    for metric in metrics:
        plot_classifier_comparison(metric, classifiers)
