"""
Creates ternary_plot, a visualization for the clusters
"""

import importlib
import pandas as pd
import plotly.graph_objects as go


def ternary_plot(df):
    """
    Creates a ternary plot
    ---

    input:  df:: pandas.Dataframe
    output:  --
    """
    # fig = px.scatter_ternary(df, a='dissuade', b='pleased', c='dominance', color='cluster', color_discrete_map={'0': 'blue', '1': 'green', '2': 'red', '3': 'pink'}, marker)

    fig = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': df['dissuade'],
        'b': df['pleased'],
        'c': df['dominance'],
        'marker': {
            'color': df['cluster'],
            'size': 15
        }
    }))

    fig.update_layout({
        'ternary':
            {
                'aaxis': {'title': 'Dissuade'},
                'baxis': {'title': 'Pleased'},
                'caxis': {'title': 'Dominance'}
            }
    }
    )


if __name__ == '__main__':
    variables = importlib.import_module('40_Realisation.code.variables')
    cluster_df = pd.read_csv(variables.getSavePath('data', '070_clustering/kmeans/cluster_n4.csv'))

    # -----------do stuff!-----------

    cluster_df = cluster_df.drop(columns=['date', 'time', 'aroused'])
    ternary_plot(cluster_df)

    print("Figure created")
    # fig.show()
    # fig.write_image(variables.getSavePath('viz', '200_custom_viz/ternary_plot.png'), scale=3)
