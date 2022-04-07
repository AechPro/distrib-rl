"""
    File name: Grapher.py
    Author: Matthew Allen

    Description:
        A helper file to enable easy use of matplotlib for graphing purposes.
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import matplotlib.cm as mplcm
import matplotlib.colors as colors

NUM_COLORS = 15

cm = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

def clear_plot():
    ax.clear()
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])

def plot_data(data,axis=None,clear=True,type='line',ylim=None):
    """
    Function to plot some data, optionally along an axis, and optionally on top of other previously
    plotted data.
    :param data: Data to plot.
    :param axis: Optional, axis over which to plot the data.
    :param clear: Optional, set to False to prevent the plot from being cleared before the data is plotted.
    :return: None.
    """

    if ylim is not None:
        plt.ylim(*ylim)

    plot_func = ax.plot
    if clear:
        clear_plot()
    if type == 'line':
        plot_func = ax.plot
    elif type == 'scatter':
        plot_func = ax.scatter

    if axis is not None:
        plot_func(axis, data)
        ticks = [axis[i] for i in range(0, len(axis), len(axis)//5)]

        if ticks[-1] != axis[-1]:
            ticks.append(axis[-1])

        ax.set_xticks(ticks)
    else:
        plot_func(data)

def set_legend(legend,loc='upper left'):
    """
    Wrapper function to set the legend of the current matplotlib plot in the lower right.
    :param legend: List of strings indicating data names.
    :return: None.
    """
    ax.legend(legend,loc=loc)

def save_plot(name, xLabel='',yLabel='',title='', path = "data/graphs"):
    """
    Function to save the current matplotlib plot.
    :param name: Name of the graph. Note that the file path is taken care of if not passed.
    :param path: Optional file path at which to save the graph.
    :return: None.
    """

    if not os.path.exists(path):
        os.makedirs(path)

    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.title(title)
    plt.savefig(''.join([path,"/graph_",name.strip().replace(" ", "_").lower()]))
    ax.clear()
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])