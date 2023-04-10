import numpy as np
import plotly.graph_objects as go

COLORS = ['blue', 'red', 'green', 'pink', 'yellow', 'violet', 'orange', 'brown', 'lightblue', 'lightgreen', 'lightpink', 'purple', 'lightcoral', 'gray', 'black']

def plot_cum_avg_reward(data, title, fig_path=None):
    _plot_figure(data, title=title, xtitle="Time step", ytitle="Cum. avg. reward", fig_path=fig_path)

def plot_perc_opt_actions(data, title, fig_path=None):
    _plot_figure(data, title=title, xtitle="Time step", ytitle="%", yrange=[0,100], fig_path=fig_path)

def _get_scatter_obj(x, y, name=None, color=None):
    if type(x) == int:
        x = np.arange(1, x+1)

    return go.Scatter(x=x, y=y, name=name, line=dict(color = COLORS[color] if color is not None else None))

def _plot_figure(data, **kwargs):

    data = [_get_scatter_obj(**data_i) for data_i in data]

    layout = go.Layout(
    title=kwargs.get('title', None)
    ,
    xaxis=dict(
        title=kwargs.get('xtitle', None),
        range=kwargs.get('xrange', None)
    ),
    yaxis=dict(
        title=kwargs.get('ytitle', None),
        range=kwargs.get('yrange', None)
    ) )

    fig = go.Figure(data=data, layout=layout)

    if kwargs.get('fig_path', None) is not None:
        fig.write_image(f"{kwargs['fig_path']}.png")

    fig.show()
