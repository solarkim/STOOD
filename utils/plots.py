import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_bar(df:pd.DataFrame,
                col_x:str, layout:tuple=None, figsize:tuple=None,
                ax=None, x_rot:int=0, legend:bool=False,
                title:str=None, title_attr:dict=None,
                bar_attr:dict=None, **kwargs) -> Axes:
    """Dataframe의 Bar 그래프

    Args:
        df (pd.DataFrame): dataframe
        col_x (str): x축 column
        layout (tuple, optional): subplot layout. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to None.
        ax (_type_, optional): 개별 axes를 할당. Defaults to None.
        x_rot (int, optional): x label rotation. Defaults to 0.
        legend (bool, optional): show legend or not. Defaults to False.
        title (str, optional): figure title. Defaults to None.
        title_attr (dict, optional): title attributes. Defaults to None.
        bar_attr (dict, optional): bar text attributes. Defaults to None.

    Returns:
        Axes: axes
    """
    fig_attr = kwargs
    fig_attr['legend'] = legend
    if ax: fig_attr['ax'] = ax
    if layout:
        fig_attr['subplots'] = True
        fig_attr['layout'] = layout
    if figsize: fig_attr['figsize'] = figsize
    
    bar_attr = bar_attr or {'size': 8}

    # plt.title(title, **title_attr)
    axes = df.plot.bar(x=col_x, rot=x_rot, title=title, **fig_attr)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    if ax:
        ax.set_title(title, **title_attr)
        
    for ax in axes:
        for container in ax.containers:
            ax.bar_label(container, **bar_attr)
    
    # plt.subplots_adjust(wspace=0.1)  # Adjust the spacing between subplots
    return ax


def show_bars(df:pd.DataFrame, x:str, title:str=None, figsize:tuple=(20,4), **kwargs):
    kwargs.update(dict(rot=45,
                    align='edge',
                    subplots=True,
                    layout=(1,5),
                    legend=False))
    axes = df.plot.bar(x=x,
                        title=title,
                        figsize=figsize,
                        **kwargs)

    for ax in axes[0]:  # Iterate over the axes in the subplot
        for p in ax.patches:  # Iterate over each bar patch
            ax.annotate(f'{p.get_height():.2f}',  # Add annotation on top of each bar
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=8,
                        color='black')

    plt.subplots_adjust(wspace=0.1)  # Adjust the spacing between subplots