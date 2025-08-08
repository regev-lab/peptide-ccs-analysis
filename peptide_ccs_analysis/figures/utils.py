import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from peptide_ccs_analysis.constants import DELTA_CCS_CMAP, GRAY, PPI


def axes_position_helper(
    figsize,
    axsize,
    left=None,
    bottom=None,
    right=None,
    top=None,
    loc=None,
    coords="margin",
    units="in",
):
    fig_width, fig_height = figsize
    ax_width, ax_height = axsize

    assert coords in ["position", "margin"]

    if coords == "margin":
        if top is not None:
            top = fig_height - top
        if right is not None:
            right = fig_width - right

    if loc is not None:
        if loc == "center":
            left = fig_width / 2 - ax_width / 2
            bottom = fig_height / 2 - ax_height / 2

        else:
            loc = loc.split(" ")
            if loc[0] == "upper":
                top = fig_height - ax_height * 0.1
            if loc[0] == "lower":
                bottom = fig_height + ax_height * 0.125
            if loc[1] == "right":
                right = fig_width - ax_width * 0.12
            if loc[1] == "left":
                left = fig_width + ax_width * 0.11

    assert bottom is not None or top is not None
    assert left is not None or right is not None

    if bottom is not None:
        top = bottom + ax_height
    else:
        bottom = top - ax_height

    if right is not None:
        left = right - ax_width
    else:
        right = left + ax_width

    return left, bottom, right, top


def compute_axes_position(
    figsize,
    axsize,
    left=None,
    bottom=None,
    right=None,
    top=None,
    loc=None,
    coords="margin",
    units="in",
):
    fig_width, fig_height = figsize

    left, bottom, right, top = axes_position_helper(
        figsize,
        axsize,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        loc=loc,
        coords=coords,
        units=units,
    )
    return [
        left / fig_width,
        bottom / fig_height,
        (right - left) / fig_width,
        (top - bottom) / fig_height,
    ]


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def margined(interval, margin=0.05):
    if interval is None:
        return None

    if not hasattr(margin, "__len__"):
        margin = [margin, margin]

    assert hasattr(margin, "__len__") and len(margin) == 2

    interval = [min(interval), max(interval)]
    ptp = np.ptp(interval)
    return [interval[0] - margin[0] * ptp, interval[1] + margin[1] * ptp]


def plot_imshow_figure(x, y, c, min_count, xlim, ylim, xlabel, ylabel, **fig_kwargs):
    temp_df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "c": c,
        }
    )

    mean_data = temp_df.pivot_table("c", "y", "x", "mean")
    mean_data = (
        mean_data
        + pd.DataFrame(  # Add empty dataframe to include possibly missing entries
            0.0,
            columns=range(min(x), max(x) + 1),
            index=range(min(y), max(y) + 1),
        )
    )

    count_data = temp_df.pivot_table("c", "y", "x", "count").replace(np.nan, 0.0)
    count_data = (
        count_data
        + pd.DataFrame(  # Add empty dataframe to include possibly missing entries
            0.0,
            columns=range(min(x), max(x) + 1),
            index=range(min(y), max(y) + 1),
        )
    )

    data = mean_data * (count_data >= min_count).astype(float).replace(0.0, np.nan)

    if "figsize" in fig_kwargs:
        figsize = fig_kwargs["figsize"]
    else:
        figsize = (6, 4)

    if "hist_height" in fig_kwargs:
        hist_height = fig_kwargs["hist_height"]
    else:
        hist_height = 0.25

    if "bin_size" in fig_kwargs:
        bin_size = fig_kwargs["bin_size"]
    else:
        bin_size = 1 / 12

    ax_width = bin_size * np.ptp(xlim)
    ax_height = bin_size * np.ptp(ylim)
    axsize = (ax_width, ax_height)
    axsize_histx = (axsize[0], hist_height)
    axsize_histy = (hist_height, axsize[1])

    ax_between = 10 / PPI  # - 0.75 * 0.1
    # cmap = 'RdBu_r'

    # center = 0.5
    # halfrange = 0.5

    # norm = mpl.colors.CenteredNorm(center, halfrange=halfrange)

    position = compute_axes_position(
        figsize, axsize, left=1, top=1 + axsize_histx[1] + ax_between, coords="margin"
    )
    position_hist_x = compute_axes_position(figsize, axsize_histx, left=1, top=1, coords="margin")
    position_hixt_y = compute_axes_position(
        figsize,
        axsize_histy,
        left=1 + ax_between + axsize[0],
        top=1 + axsize_histx[1] + ax_between,
        coords="margin",
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.gcf().add_axes(position)
    ax_histx = plt.gcf().add_axes(position_hist_x)
    ax_histy = plt.gcf().add_axes(position_hixt_y)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines['bottom'].set_bounds([0,1])
    # ax.spines['left'].set_bounds([0,1])

    ax_histx.spines["top"].set_visible(False)
    # ax_histx.spines['bottom'].set_visible(False)
    ax_histx.spines["left"].set_visible(False)
    ax_histx.spines["right"].set_visible(False)
    # ax_histx.spines['bottom'].set_position(('outward', 0.05 * PPI))
    # ax_histx.spines['bottom'].set_bounds([0,1])
    # ax_histx.spines['left'].set_bounds([0,1])
    ax_histx.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        # bottom=False,
        left=False,
        top=False,
        right=False,
    )

    ax_histy.spines["top"].set_visible(False)
    ax_histy.spines["bottom"].set_visible(False)
    # ax_histy.spines['left'].set_visible(False)
    ax_histy.spines["right"].set_visible(False)
    # ax_histy.spines['left'].set_position(('outward', 0.05 * PPI))
    # ax_histy.spines['bottom'].set_bounds([0,1])
    # ax_histy.spines['left'].set_bounds([0,1])
    ax_histy.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        # left=False,
        top=False,
        right=False,
    )

    plt.sca(ax)

    plt.imshow(
        data,
        origin="lower",
        # cmap='RdBu_r',
        # cmap='RdYlBu_r',
        cmap=DELTA_CCS_CMAP,
        # cmap='viridis',
        vmin=0,
        vmax=1,
        extent=(min(x) - 0.5, max(x) + 0.5, min(y) - 0.5, max(y) + 0.5),
        aspect="auto",
    )

    xlim_delta = 0.4 / PPI * np.ptp(xlim) / ax_width
    ylim_delta = 0.4 / PPI * np.ptp(ylim) / ax_height

    plt.xlim([xlim[0] - xlim_delta, xlim[1]])
    plt.ylim([ylim[0] - ylim_delta, ylim[1]])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.sca(ax_histx)
    plt.hist(
        x,
        bins=np.arange(xlim[0], xlim[1] + 1),
        # color=lighten_color(GRAY, 0.5),
        # edgecolor=GRAY,
        color=lighten_color(GRAY, 0.25),
        edgecolor=[lighten_color(GRAY, 0.5)],
        linewidth=0.8,
    )
    # sns.kdeplot(x, color=GRAY)
    plt.xticks(ax.get_xticks())
    plt.xlim(ax.get_xlim())
    # plt.yscale('log')

    plt.sca(ax_histy)
    plt.hist(
        y,
        bins=np.arange(ylim[0], ylim[1] + 1),
        orientation="horizontal",
        # color=lighten_color(GRAY, 0.5),
        # edgecolor=GRAY,
        color=lighten_color(GRAY, 0.25),
        edgecolor=[lighten_color(GRAY, 0.5)],
        linewidth=0.8,
    )
    # plt.twiny()
    # sns.kdeplot(y, vertical=True, color='k',zorder=99)
    plt.yticks(ax.get_yticks())
    plt.ylim(ax.get_ylim())
    # plt.xscale('log')

    return fig, [ax, ax_histx, ax_histy, plt.gca()]
