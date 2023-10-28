from datetime import datetime, timedelta

import argparse

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec

from matplotlib.pylab import rcParams

from scipy import stats




def generate_histogram(annotations, dt, filtersize=10):
    # generate a histogram that contains the percentagewise presence of the
    # activity given a time frame
    annotations['date'] = [x.split()[0] for x in annotations['datetime']]
    annotations['time'] = [x.split()[1] for x in annotations['datetime']]

    histograms = []
    histogram_keys = []

    td = timedelta(minutes=int(dt))
    ts = np.arange(datetime(1900, 1, 1, 0, 0, 0), datetime(1900, 1, 1, 23, 59, 59), td).astype(datetime)
    tsstr = [x.strftime("%H:%M:%S") for x in ts]

    for i, t in enumerate(tsstr[:-1]):
        df1 = annotations[(annotations['time'] >= t) & (annotations['time'] < tsstr[i + 1])]

        # reject any frame that is annotated less than x times
        daysfilter = df1.groupby(['date']).size().apply(lambda x: x > filtersize)
        daysfilter = list(daysfilter[daysfilter].index)
        df1 = df1[df1['date'].isin(daysfilter)]
        if len(df1):
            df1 = pd.crosstab(df1['date'], df1['variable']).apply(lambda r: r / r.sum(), axis=1)
            if len(df1) > 0:
                histograms.append(df1)
                histogram_keys.append(t)

    h_all = pd.DataFrame()

    for i, h in enumerate(histograms):
        h['time'] = histogram_keys[i]
        h_all = pd.concat(histograms)

    h_all.replace(np.nan, 0, inplace=True)

    return h_all


def parallel_coordinates_for_multinomial_distribution(
        df, group_column_name, class_names, class_colors=None,
        col_width=0.8,
        max_prob=None, max_prob_rounding=5,
        sub_group_column_name=None,
        class_cmap='tab10',
        sub_group_cmap='Pastel1',
        ax=None,
        kw_lines={
            'color': 'gray',
            'alpha': 0.2,
            'lw': 1,
            'zorder': 0},
        row2linestyle=None,
        kw_background={
            'color': '0.95',
            'zorder': 0
        },
        show_violins=True,
        violins_cmeans_lw=5,
        violin_use_mean4width=True,
        violin_width=1,
        violin_zorder=5,
        kw_violins={
            'showmeans': True,
            'showmedians': False,
            'showextrema': False,
        },
        prob_anno_count=5,
        kw_prob_annotations={  # parameters passed to the annotation text for probabilities
            'color': ".5",
            'rotation': 'vertical',
            'horizontalalignment': 'center',
            'verticalalignment': 'bottom',
            'zorder': 1
        },
        class_annotations={
            'alpha': 0.1,
            'zorder': 0,
            'lw': 1,
            'ls': '--'
        },
        filename=None):
    """
    Create a plot in parallel coordinates for a multi-class distribution.
    """
    groups = df[group_column_name].unique()
    groups.sort()

    group2x = {g: i for i, g in enumerate(groups)}

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(len(groups) * 3, len(class_names)))
        plt.tight_layout()

    if max_prob is None:
        max_prob = df[class_names].max().max()
        max_prob_f = 100 / max_prob_rounding
        max_prob = np.ceil(max_prob * max_prob_f) / max_prob_f

    ys = range(len(class_names))

    if class_colors is None:
        cmap = get_cmap(class_cmap)
        class_colors = [cmap.colors[i] for i, _c in enumerate(class_names)]

    sub_group2color = {}
    if sub_group_column_name is not None:
        sub_groups = df[sub_group_column_name].unique()
        sub_groups.sort()

        cmap = get_cmap(sub_group_cmap)
        sub_group2color = {g: cmap.colors[i] for i, g in enumerate(sub_groups)}

    xs = col_width / 2 + np.array(list(group2x.values()))

    print(f'Data will be displayed in {len(groups)} groups: {groups}')

    assert np.isclose(df[class_names].sum(axis=1),
                      1).all(), f"For each row in the dataframe the columns {class_names} must sum up to 1.0"

    ymargin = 0.5
    ymin = 0 - ymargin
    ytext = len(ys) - 1 + ymargin
    ymax = ytext

    if prob_anno_count > 0:
        ymax += 0.5

    ax.set_ylim(ymin, ymax)

    if show_violins:
        for g in groups:
            widths = [1.0] * len(class_names)

            if violin_use_mean4width:
                widths = 2 * df[df[group_column_name] == g][class_names].mean()

            width = list(widths / max(widths) * violin_width)

            d4v = [list(group2x[g] + col_width / max_prob * np.array(df[df[group_column_name] == g][c])) for c in
                   class_names]
            print(d4v)
            vps = ax.violinplot(d4v, vert=False, positions=ys, widths=widths, **kw_violins)

            for c, vpb in zip(class_colors, vps['bodies']):
                vpb.set_facecolor(c)
                vpb.set_zorder(5)

            for dn in ['cmeans', 'cmins', 'cmaxes', 'cbars', 'cmedians']:
                if vps.get(dn):
                    vps[dn].set_colors(class_colors)
                    vps[dn].set_zorder(5)

            if vps.get('cmeans'):
                vps.get('cmeans').set_linewidth(violins_cmeans_lw)
                vps.get('cmeans').set_zorder(5)

    # plot a line for each row in the dataset
    for i, row in df.iterrows():
        xvs = group2x[row[group_column_name]] + col_width / max_prob * np.array([row[c] for c in class_names])

        if sub_group_column_name:
            kw_lines['color'] = sub_group2color.get(row[sub_group_column_name])

        if row2linestyle:
            kw_lines.update(row2linestyle(row))

        ax.plot(xvs, ys, **kw_lines)

    # show backgrounds
    for g, x in group2x.items():
        ax.fill_betweenx([ymin, ymax], [x] * 2, [x + col_width] * 2, **kw_background)

        if prob_anno_count > 0:
            for p in np.linspace(0, max_prob, prob_anno_count):
                xx = x + p * col_width / max_prob
                if p > 0 and p < max_prob:
                    ax.axvline(xx, ymax=ytext / ymax, color='gray', lw=1, ls=':', zorder=0)
                    ax.annotate('%2.0f%%' % (p * 100), (xx, ytext), **kw_prob_annotations)
                else:
                    ax.axvline(xx, color='gray', lw=1, ls=':', zorder=0)

    for y, c in enumerate(class_colors):
        ax.axhline(y, color=c, **class_annotations)

    ax.set_yticks(ys)
    ax.set_yticklabels(class_names)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)

    if filename:
        plt.savefig(filename, dpi=200)


def r2ls(row):
    if row['day'] > 5: return {'color': 'red'}
    return {}
