# Basic
import os
import sys
sys.path.insert(0, "..")

import umap
import numpy as np
import pandas as pd

# For visualization
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from helpers import read_json

def visualize_eyemovements(data: pd.DataFrame, fn: str,
                           x_col: str="x", y_col: str="y",
                           time_col: str='timestamps',
                           color: str="movement_name"):
    assert ((x_col in data.columns) and (y_col in data.columns)
            and (color in data.columns) and (time_col in data.columns))

    color_mapping = dict(read_json(os.path.join(sys.path[0], "settings", "color_mappings.json")))
    data['color'] = data[color].apply(lambda x: color_mapping.get(x, "black"))
    names_mapping = dict(read_json(os.path.join(sys.path[0], "settings", "eng_rus_names.json")))
    data['rus_movements'] = data[color].apply(lambda x: names_mapping.get(x, "black"))

    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.06,
        specs=[[{}, {}],
               [{"colspan": 2}, None]],
        row_heights=[0.4, 0.6],
        subplot_titles=("Взгляд, координата Х", "Взгляд, координата Y", "Взгляд, координаты X-Y")
    )

    min_ts = np.min(data[time_col])
    for movement_type, df in data.groupby(by='rus_movements'):
        fig.add_trace(go.Scatter(x=df[time_col] - min_ts,
                                 y=df[x_col],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type,
                                 showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=df[time_col] - min_ts,
                                 y=df[y_col],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type,
                                 showlegend=False), row=1, col=2)

        fig.add_trace(go.Scatter(x=df[x_col],
                                 y=df[y_col],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type), row=2, col=1)

    fig.update_traces(mode='markers', marker_line_width=0.1, marker_size=4)
    fig.update_layout(height=800, width=1000,
                      title_text="Классификация движений глаз")

    fig.update_layout(legend_title_text='Типы движений глаз',
                      legend=dict(font=dict(family="Arial", size=12)))
    fig.update_layout(showlegend=True)

    plotly.offline.plot(fig, filename='../output/'+ fn + '.html')


def visualize_quality(y_true, y_pred, y_pred_probas):
    _plot_roc_curve(y_true, y_pred_probas)
    _plot_confusion_matrix(y_true, y_pred)


def _plot_roc_curve(y_true, y_pred_probas):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probas)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.update_layout(
        autosize=False,
        width=500,
        height=600,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4))
    plotly.offline.plot(fig, filename='../output/roc_curve.html')


def _plot_confusion_matrix(y_true, y_pred):
    cc = confusion_matrix(y_true, y_pred)
    cc = cc[::-1]

    x = ['IMPOSTORS', 'GENUINE USERS']
    y = x[::-1].copy()  # invert idx values of x

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in cc]

    # set up figure
    fig = ff.create_annotated_heatmap(cc, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5, y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper", yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.2, y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper", yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=150))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4))
    plotly.offline.plot(fig, filename='../output/confusion_matrix.html')


def reduce_dim_embeddings_UMAP(embeddings: np.ndarray,
                               n_neighbors: int=15, dim=2):
    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                           min_dist=0., n_components=dim)
    return umap_model.fit_transform(embeddings)


def plot_embeddings_2D(embeddings: np.ndarray, targets: np.ndarray):
    name_mapping = {"1": "Движения верифицируемого", "0": "Движения иных людей"}

    fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1],
                     color=list(map(lambda x: name_mapping.get(x), targets)))
    fig.update_traces(mode='markers', marker_line_width=0.1, marker_size=7)
    fig.update_layout(height=600, width=1000,
                      title_text="Верификация по следящим движениям глаз")

    fig.update_layout(legend_title_text='Принадлежность движений',
                      legend=dict(font=dict(family="Arial", size=12)))
    fig.update_layout(showlegend=True)
    plotly.offline.plot(fig, filename='../output/embeddings.html')