# Basic
import datetime
import os
import sys
from pathlib import Path

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
from config import config

import logging_handler
logger = logging_handler.get_logger(__name__)


def visualize_eyemovements(data: pd.DataFrame,
                           to_save: bool = False, session_num: int = 0):
    """
    Visualize in simple scatter plot eye movements classification results.
    Input data - is a DataFrame with time, x, y and classification labels
    """
    if not (("gaze_X" in data.columns) and ("gaze_Y" in data.columns)):
        logger.error(f"Gaze X & Y columns do not exists in given data:\n{data.columns}")
        raise AttributeError

    if not ("movements_type" in data.columns):
        logger.error(f"`movements_type` column do not exists in given data:\n{data.columns}")
        raise AttributeError

    if not ("timestamps" in data.columns):
        logger.error(f"`timestamps` column do not exists in given data:\n{data.columns}")
        raise AttributeError

    settings_dir = Path(config.get("Basic", "settings_dir"))
    if not settings_dir.exists():
        logger.error(f"Settings dir do not exist by path: {settings_dir}")
        raise FileNotFoundError

    if not (settings_dir / "color_mappings.json").exists() or not (settings_dir / "eng_rus_names.json").exists():
        logger.error(f"Settings files for visualizations do not exist by path: {settings_dir}")
        raise FileNotFoundError

    color_mapping = dict(read_json(str(settings_dir / "color_mappings.json")))
    data['color'] = data["movements_type"].apply(lambda x: color_mapping.get(x, "black"))
    names_mapping = dict(read_json(str(settings_dir / "eng_rus_names.json")))
    data['rus_movements'] = data["movements_type"].apply(lambda x: names_mapping.get(x, "black"))

    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.06,
        specs=[[{}, {}],
               [{"colspan": 2}, None]],
        row_heights=[0.4, 0.6],
        subplot_titles=("Взгляд, координата Х", "Взгляд, координата Y", "Взгляд, координаты X-Y")
    )

    min_ts = np.min(data["timestamps"])
    for movement_type, df in data.groupby(by='rus_movements'):
        fig.add_trace(go.Scatter(x=df["timestamps"] - min_ts,
                                 y=df["gaze_X"],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type,
                                 showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=df["timestamps"] - min_ts,
                                 y=df["gaze_Y"],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type,
                                 showlegend=False), row=1, col=2)

        fig.add_trace(go.Scatter(x=df["gaze_X"],
                                 y=df["gaze_Y"],
                                 mode='markers',
                                 marker_color=df['color'],
                                 name=movement_type), row=2, col=1)

    fig.update_traces(mode='markers', marker_line_width=0.1, marker_size=4)
    fig.update_layout(height=800, width=1000,
                      title_text="Классификация движений глаз")

    fig.update_layout(legend_title_text='Типы движений глаз',
                      legend=dict(font=dict(family="Arial", size=12)))
    fig.update_layout(showlegend=True)

    if to_save:
        if not Path(config.get("Basic", "output_dir")).exists():
            logger.info(f"Output dir is not exists. Creating at {config.get('Basic', 'output_dir')}...")
            Path(config.get("Basic", "output_dir")).mkdir(parents=True, exist_ok=True)

        fn = f"eyemovements_visualization_from_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}.html"
        # If file with this name already exists -> previous session of current experiment
        if (Path(config.get("Basic", "output_dir")) / fn).exists():
            fn = fn.split(".")[0] + f"_sess_{session_num}" + fn.split(".")[-1]
        fig.write_html(file=str(Path(config.get("Basic", "output_dir")) / fn), include_plotlyjs=False)
        fig.show()
        logger.info(f"Visualizations file successfully saved to: {str(Path(config.get('Basic', 'output_dir')) / fn)}")
    else:
        fig.show()


def visualize_quality(y_true, y_pred, y_pred_probas):
    _plot_roc_curve(y_true, y_pred_probas)
    _plot_confusion_matrix(y_true, y_pred)


def _plot_roc_curve(y_true, y_pred_probas):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probas)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Кривая (AUC={auc(fpr, tpr):.4f})',
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

    if ~Path(config.get("Basic", "output_dir")).exists():
        logger.info(f"Output dir is nor exists. Creating at {config.get('Basic', 'output_dir')}...")
        Path(config.get("Basic", "output_dir")).mkdir(parents=True, exist_ok=True)

    plotly.offline.plot(fig, filename=str(Path(config.get("Basic", "output_dir")) / 'roc_curve.html'))


def _plot_confusion_matrix(y_true, y_pred):
    cc = confusion_matrix(y_true, y_pred)
    cc = cc[::-1]

    x = ['Другие', 'Верифицируемый']
    y = x[::-1].copy()  # invert idx values of x

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in cc]

    # set up figure
    fig = ff.create_annotated_heatmap(cc, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Матрица ошибок</b></i>')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5, y=-0.15,
                            showarrow=False,
                            text="Предсказания",
                            xref="paper", yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.2, y=0.5,
                            showarrow=False,
                            text="Истинные",
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

    if ~Path(config.get("Basic", "output_dir")).exists():
        logger.info(f"Output dir is nor exists. Creating at {config.get('Basic', 'output_dir')}...")
        Path(config.get("Basic", "output_dir")).mkdir(parents=True, exist_ok=True)

    plotly.offline.plot(fig, filename=str(Path(config.get("Basic", "output_dir")) / 'confusion_matrix.html'))


def reduce_dim_embeddings_UMAP(embeddings: np.ndarray,
                               n_neighbors: int=15, dim=2):
    # todo: move to separate file
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

    if ~Path(config.get("Basic", "output_dir")).exists():
        logger.info(f"Output dir is nor exists. Creating at {config.get('Basic', 'output_dir')}...")
        Path(config.get("Basic", "output_dir")).mkdir(parents=True, exist_ok=True)

    plotly.offline.plot(fig, filename=str(Path(config.get("Basic", "output_dir")) / 'embeddings.html'))