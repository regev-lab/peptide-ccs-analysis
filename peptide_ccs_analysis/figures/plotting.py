from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from peptide_ccs_analysis import load_data
from peptide_ccs_analysis.constants import (
    AA_VOCABULARY,
    DENSITY_CMAP,
    DISCRETIZED_SEPARATION_CMAP,
    GAM_SPLINE_LOWER_MODE_COLOR,
    GAM_SPLINE_UPPER_MODE_COLOR,
    GRAY,
    HEXBIN_XLIM,
    HEXBIN_YLIM,
    PPI,
    SCATTER_COLOR,
)
from peptide_ccs_analysis.figures.utils import (
    compute_axes_position,
    lighten_color,
    margined,
    plot_imshow_figure,
)
from peptide_ccs_analysis.models.loading import load_model


def setup_rcParams():
    mpl.rcdefaults()
    FONT = "Arial"
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.size"] = 8.33
    mpl.rcParams["font.family"] = ["Arial"]
    mpl.rcParams["font.sans-serif"] = [FONT]
    mpl.rcParams["savefig.transparent"] = True
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.bf"] = "Arial:bold"


setup_rcParams()


def plot_figure_1a_left():
    DATA1 = load_data.DATA1()

    figsize = (4, 4)
    hist_height = 0.25
    axsize = (1.25, 1.25)
    axsize_histx = (axsize[0], hist_height)
    axsize_histy = (hist_height, axsize[1])

    ax_between = 10 / PPI
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

    plt.figure(figsize=figsize)
    ax = plt.gcf().add_axes(position)
    ax_histx = plt.gcf().add_axes(position_hist_x)
    ax_histy = plt.gcf().add_axes(position_hixt_y)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_histx.spines["top"].set_visible(False)
    ax_histx.spines["left"].set_visible(False)
    ax_histx.spines["right"].set_visible(False)

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
    ax_histy.spines["right"].set_visible(False)
    ax_histy.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        top=False,
        right=False,
    )

    plt.sca(ax)
    df = DATA1.query("charge==3")

    nx = 75

    ny = int(nx * axsize[1] / axsize[0] / np.sqrt(3))
    image = plt.hexbin(
        df["mass"],
        df["ccs"],
        edgecolors="none",
        mincnt=1,
        cmap=DENSITY_CMAP,
        gridsize=(nx, ny),
        alpha=1,
    )

    plt.xlim(HEXBIN_XLIM)
    plt.ylim(HEXBIN_YLIM)
    plt.xlabel("Mass (Da)")
    plt.ylabel(r"CCS ($\mathrm{\AA}^2$)")

    plt.sca(ax_histx)
    sns.kdeplot(df["mass"], color=GRAY)
    plt.xticks(ax.get_xticks())
    plt.xlim(ax.get_xlim())

    plt.sca(ax_histy)
    sns.kdeplot(y=df["ccs"], color=GRAY)
    plt.yticks(ax.get_yticks())
    plt.ylim(ax.get_ylim())

    plt.show(block=False)

    counts = image.get_array()
    figsize = (2, 2)
    cbar_width = 1
    axsize = (cbar_width, cbar_width * 0.075 / 0.77)
    position = compute_axes_position(figsize, axsize, bottom=0.5, left=0.15, coords="margin")

    plt.figure(figsize=figsize)
    a = np.array([[0, max(counts)]])
    plt.imshow(a, cmap=DENSITY_CMAP)
    plt.gca().set_visible(False)
    plt.colorbar(orientation="horizontal")
    cax = plt.gca().images[-1].colorbar.ax
    cax.set_position(position)
    cax.xaxis.set_label_position("top")
    cax.set_xlabel("Bin count", fontsize="small")
    cax.tick_params(labelsize="small")
    plt.show(block=False)


def plot_figure_1a_right():
    DATA1 = load_data.DATA1()

    figsize = (4, 4)
    axsize = (0.75, 0.75)

    ax_between = 10 / PPI
    position = compute_axes_position(
        figsize, axsize, left=1, top=1.25 + ax_between, coords="margin"
    )

    plt.figure(figsize=figsize)
    ax = plt.gcf().add_axes(position)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.sca(ax)
    df = DATA1.query("charge==3")

    nx = 50
    ny = int(nx * axsize[1] / axsize[0] / np.sqrt(3))

    plt.hexbin(
        df["mass"],
        df["ccs"],
        C=df["delta_ccs"] > 0,
        reduce_C_function=lambda array: np.mean(array) > 0.5,
        edgecolors="none",
        mincnt=1,
        cmap=DISCRETIZED_SEPARATION_CMAP,
        gridsize=(nx, ny),
        alpha=1,
    )

    plt.xlim(HEXBIN_XLIM)
    plt.ylim(HEXBIN_YLIM)

    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.show(block=False)


def plot_figure_1b():
    DATA1 = load_data.DATA1()

    for k in [2, 3, 4]:
        figsize = (4, 4)
        axsize = (0.9, 0.9)

        position = compute_axes_position(
            figsize, axsize, left=1, top=1.25 + 10 / PPI, coords="margin"
        )

        plt.figure(figsize=figsize)
        ax = plt.gcf().add_axes(position)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.sca(ax)
        df = DATA1.query("charge==3")
        df = df.query(f"basic_site_count=={k}")

        nx = 50

        ny = int(nx * axsize[1] / axsize[0] / np.sqrt(3))
        plt.hexbin(
            df["mass"],
            df["ccs"],
            edgecolors="none",
            mincnt=1,
            cmap=DENSITY_CMAP,
            gridsize=(nx, ny),
            alpha=1,
        )

        plt.xlim(HEXBIN_XLIM)
        plt.ylim(HEXBIN_YLIM)
        plt.xlabel("Mass (Da)")

        if k in [2]:
            plt.ylabel(r"CCS ($\mathrm{\AA}^2$)")

        if k in [3, 4]:
            plt.gca().set_yticklabels([])

        plt.show(block=False)


def plot_figure_1c():
    DATA1 = load_data.DATA1()

    xlabel = "Distance of internal basic site to C-terminus"
    ylabel = "Peptide length"

    xlim = [1 - 0.5, 30 + 0.5]
    ylim = [15 - 0.5, 30 + 0.5]

    min_count = 8

    df = DATA1.query("charge==3 & basic_site_count==3")
    df = df.query("len >= 15")
    df = df.query("not is_acetylated")
    df = df[df["sequence"].apply(lambda x: x[-1] in "KR")]

    x = df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1])
    y = df["len"]
    c = df["delta_ccs"] > 0

    idx = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])

    df = df[idx]
    x = x[idx]
    y = y[idx]
    c = c[idx]

    plot_imshow_figure(
        x=x,
        y=y,
        c=c,
        min_count=min_count,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    plt.show(block=False)


def plot_figure_1d():
    DATA1 = load_data.DATA1()

    xlim = [0 - 0.5, 5 + 0.5]
    ylim = [15 - 0.5, 30 + 0.5]
    min_count = 10

    for key in ["dist_to_c_term_leq_5", "6_leq_dist_to_cd_term_lt_7", "dist_to_cd_term_geq_8"]:
        df = DATA1.query("charge==3 & basic_site_count==3")
        df = df.query("len >= 15")
        df = df.query("not is_acetylated")
        df = df[df["sequence"].apply(lambda x: x[-1] in "KR")]

        if key == "dist_to_c_term_leq_5":
            df = df[df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1]) <= 5]

        elif key == "6_leq_dist_to_cd_term_lt_7":
            df = df[
                (df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1])).apply(
                    lambda x: 6 <= x <= 7
                )
            ]
        elif key == "dist_to_cd_term_geq_8":
            df = df[df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1]) >= 8]

        x = df["sequence"].str.count("P")
        y = df["len"]
        c = df["delta_ccs"] > 0

        xlabel = "Proline count"
        ylabel = "Peptide length"

        idx = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])

        df = df[idx]
        x = x[idx]
        y = y[idx]
        c = c[idx]

        fig, axes = plot_imshow_figure(
            x=x,
            y=y,
            c=c,
            min_count=min_count,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        if key != "dist_to_c_term_leq_5":
            plt.sca(axes[0])
            plt.ylabel("")
            plt.gca().set_yticklabels([])

        plt.show(block=False)


def plot_figure_1f():
    SIMULATION_DATA = load_data.SIMULATION_DATA()
    SIMULATION_PEPTIDES = load_data.SIMULATION_PEPTIDES()

    simulation_peptides = [
        SIMULATION_PEPTIDES["Gpos_1"][0],
        SIMULATION_PEPTIDES["Gpos_2"][2],
    ]

    data = [
        (
            SIMULATION_DATA["Gpos_1_mean_helix_content_vs_residue"]["residue"],
            SIMULATION_DATA["Gpos_1_mean_helix_content_vs_residue"]["position_0"],
        ),
        (
            SIMULATION_DATA["Gpos_2_mean_helix_content_vs_residue"]["residue"],
            SIMULATION_DATA["Gpos_2_mean_helix_content_vs_residue"]["position_2"],
        ),
    ]

    figsize = (8, 11)
    plt.figure(figsize=figsize)

    axsize = (1.2, 0.75)
    between_spacing = 0.12
    vertical_between_spacing = 0.24

    n_rows = 1
    n_cols = 2
    N = len(simulation_peptides)

    axes_grid_positions = [
        compute_axes_position(
            figsize,
            axsize,
            left=1 + j * (axsize[0] + between_spacing),
            top=1 + i * (axsize[1] + vertical_between_spacing),
        )
        for i in range(n_rows)
        for j in range(n_cols)
        if i * n_cols + j < N
    ]

    axes_grid = []
    for position in axes_grid_positions:
        ax = plt.gcf().add_axes(position)
        axes_grid.append(ax)

    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j < N:
                if j > 0:
                    axes_grid[i * n_cols + j].get_yaxis().set_ticklabels([])
                if i * n_cols + j + n_cols < N:
                    axes_grid[i * n_cols + j].get_xaxis().set_ticklabels([])

    grid_left = 1
    grid_right = 1 + n_cols * (axsize[0] + between_spacing) - between_spacing
    grid_bottom = 1 + n_rows * (axsize[1] + vertical_between_spacing) - vertical_between_spacing

    axes_grid[0].set_ylabel(r"$\alpha$-Helix fraction", labelpad=8)

    plt.figtext(
        (grid_left + grid_right) / 2 / figsize[0],
        1 - (grid_bottom + 15 / PPI) / figsize[1],
        "Residue position",
        ha="center",
        va="top",
    )

    for i, ax in zip(range(N), axes_grid):
        plt.sca(ax)
        plt.plot(data[i][0], data[i][1], ".-")
        plt.ylim([0, 1])

        plt.annotate(
            simulation_peptides[i],
            xy=(0.5, 1),
            xytext=(0, 0.3),
            xycoords="axes fraction",
            textcoords="offset fontsize",
            va="bottom",
            ha="center",
            fontsize="small",
        )

        plt.gca().yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

        plt.gca().xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        plt.gca().xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

    plt.show(block=False)


def plot_figure_2():
    DATA1 = load_data.DATA1()

    xlabel = "Distance of second internal\nbasic site to C-terminus"
    ylabel = "Distance of first internal\nbasic site to C-terminus"

    xlim = [1 - 0.5, 30 + 0.5]
    ylim = [1 - 0.5, 30 + 0.5]

    min_count = 8

    for tag in ["all", "RK_H"]:
        print(tag)
        df = DATA1.query("charge==3 & basic_site_count==4")
        df = df.query("len >= 15")
        df = df.query("not is_acetylated")

        df = df[df["sequence"].apply(lambda x: x[-1] in "RK")]

        match tag:
            case "all":
                pass
            case "RK_H":
                df = df[
                    df["basic_sites"].apply(
                        lambda x: x[1] in "RK" and x[2] in "H" and x[3] in "RK"
                    )
                ]

        x = df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[2])
        y = df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1])
        c = df["delta_ccs"] > 0

        idx = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])

        df = df[idx]
        x = x[idx]
        y = y[idx]
        c = c[idx]

        fig, axes = plot_imshow_figure(
            x=x,
            y=y,
            c=c,
            min_count=min_count,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
            bin_size=1 / 24,
        )

        plt.show(block=False)


def plot_figure_3_and_4abc():
    amino_acid_order = [
        "A",
        "V",
        "I",
        "L",
        "F",
        "M",
        "Y",
        "W",
        "S",
        "T",
        "G",
        "Q",
        "N",
        "E",
        "D",
        "P",
        "C",
    ]

    feature_order = amino_acid_order + ["R(?!$)", "K(?!$)", "H"]

    feature_label_dict = {feature: feature for feature in feature_order}
    feature_label_dict.update(
        {
            "C": r"C$^\dag$",
            "R(?!$)": r"R$^\mathbf{∗}$",
            "K(?!$)": r"K$^\mathbf{∗}$",
            "len": "length",
            "ind_K$": "C-term K",
        }
    )

    colors = [
        GAM_SPLINE_LOWER_MODE_COLOR,
        GAM_SPLINE_UPPER_MODE_COLOR,
    ]

    x_offsets = [-0.15, 0.15]  # x-offsets for the C-term indicator boxplot

    figure_keys = [
        "intermode",
        "intramode",
    ]
    figure_params_dict = {
        "intermode": {},
        "intramode": {},
    }

    model_names = [
        "CCSModeSharedSplineGAM_InternalBasicSiteAwayFromCTerm",
        "CCSModeSharedSplineGAM_InternalBasicSiteNearCTerm",
    ]
    ylim = margined([-2, 2])
    ylabel = "Contribution to higher CCS mode (logit)"
    figure_params_dict["intermode"]["model_names"] = model_names
    figure_params_dict["intermode"]["ylim"] = ylim
    figure_params_dict["intermode"]["ylabel"] = ylabel

    model_names = [
        "AbsoluteCCSSharedSplineGAM_InternalBasicSiteAwayFromCTermLowMode",
        "AbsoluteCCSSharedSplineGAM_InternalBasicSiteNearCTermHighMode",
    ]
    ylim = margined([-25 / 2, 25 / 2])
    ylabel = r"Contribution to absolute CCS ($\mathrm{\AA}^2$)"
    figure_params_dict["intramode"]["model_names"] = model_names
    figure_params_dict["intramode"]["ylim"] = ylim
    figure_params_dict["intramode"]["ylabel"] = ylabel

    for figure_key in figure_keys:
        figure_params = figure_params_dict[figure_key]
        model_names = figure_params["model_names"]
        ylim = figure_params["ylim"]
        ylabel = figure_params["ylabel"]
        model_dict = {model_name: load_model(model_name) for model_name in model_names}

        figsize = (8, 11)
        plt.figure(figsize=figsize)

        axsize = (0.70, 0.75)
        between_spacing = 0.10
        large_vertical_between_spacing = 6 * between_spacing
        large_horizontal_between_spacing = 8 * between_spacing
        medium_horizontal_between_spacing = 8 * between_spacing

        n_rows = 4
        n_cols = 5

        axes_grid_positions = [
            compute_axes_position(
                figsize,
                axsize,
                left=1 + j * (axsize[0] + between_spacing),
                top=1 + i * (axsize[1] + between_spacing),
            )
            for i in range(n_rows)
            for j in range(n_cols)
        ]

        axes_grid = []
        for position in axes_grid_positions:
            ax = plt.gcf().add_axes(position)
            axes_grid.append(ax)

        for i in range(n_rows):
            for j in range(n_cols):
                if j > 0:
                    axes_grid[i * n_cols + j].get_yaxis().set_ticklabels([])
                if i < n_rows - 1:
                    axes_grid[i * n_cols + j].get_xaxis().set_ticklabels([])

        # These define the position (in inches from the top left of the figure)
        # for the different sides of the grid of axes that are used to plot
        # the amino acid splines
        grid_left = 1
        grid_right = 1 + n_cols * (axsize[0] + between_spacing) - between_spacing
        grid_top = 1
        grid_bottom = 1 + n_rows * (axsize[1] + between_spacing) - between_spacing

        plt.figtext(
            (grid_left - 20 / PPI) / figsize[0],
            1 - (grid_top + grid_bottom) / 2 / figsize[1],
            ylabel,
            ha="right",
            va="center",
            rotation=90,
            fontsize="medium",
        )

        plt.figtext(
            (grid_left + grid_right) / 2 / figsize[0],
            1 - (grid_bottom + 18.6 / PPI) / figsize[1],
            "Relative position",
            ha="center",
            va="top",
        )

        len_ind_top = grid_top + 7 * between_spacing
        len_ind_bottom = len_ind_top + axsize[1]
        len_ind_right = (
            grid_right + large_horizontal_between_spacing + 2 * axsize[0] + between_spacing + 0.24
        )

        position = compute_axes_position(
            figsize,
            axsize,
            left=grid_right + large_horizontal_between_spacing,
            top=len_ind_top,
        )

        ax_len = plt.gcf().add_axes(position)
        plt.xlabel("Peptide length", labelpad=2)

        position = compute_axes_position(
            figsize,
            axsize,
            left=len_ind_right - axsize[0],
            top=len_ind_top,
        )

        ax_ind = plt.gcf().add_axes(position)
        plt.xlabel("C-terminal\namino acid", labelpad=2)

        scatter_top = len_ind_bottom + large_vertical_between_spacing
        scatter_bottom = grid_bottom
        scatter_left = grid_right + medium_horizontal_between_spacing
        scatter_right = len_ind_right
        position = compute_axes_position(
            figsize,
            (scatter_right - scatter_left, scatter_bottom - scatter_top),
            left=scatter_left,
            top=scatter_top,
        )

        ax_scatter = plt.gcf().add_axes(position)

        match figure_key:
            case "intermode":
                ylabel = ["Contribution to the", "higher CCS mode (logit)"]
            case "intramode":
                ylabel = ["Contribution to", r"absolute CCS ($\mathrm{\AA}^2$)"]

        for i, line in enumerate(ylabel[::-1]):
            offset = 10 / PPI
            plt.figtext(
                (scatter_left - 32 / PPI - offset * i) / figsize[0],
                1 - (len_ind_top + len_ind_bottom) / 2 / figsize[1],
                line,
                ha="center",
                va="center",
                rotation=90,
                fontsize="medium",
            )

        match figure_key:
            case "intermode":
                ylabel = ["Average contribution to", "the higher CCS mode (logit)"]
            case "intramode":
                ylabel = ["Average contribution", r"to absolute CCS ($\mathrm{\AA}^2$)"]

        for i, line in enumerate(ylabel[::-1]):
            offset = 10 / PPI
            plt.figtext(
                (scatter_left - 32 / PPI - offset * i) / figsize[0],
                1 - (scatter_top + scatter_bottom) / 2 / figsize[1],
                line,
                ha="center",
                va="center",
                rotation=90,
                fontsize="medium",
            )

        for model_name, color in zip(model_names, colors):
            model = model_dict[model_name]

            multi_spline_axes = []

            for ax, feature in zip(axes_grid + [ax_len], feature_order + ["len"]):
                plt.sca(ax)

                if not feature == "len":
                    plt.annotate(
                        feature_label_dict[feature],
                        xy=(1, 0),
                        xytext=(-1, 0.3),
                        xycoords="axes fraction",
                        textcoords="offset fontsize",
                        va="bottom",
                        ha="right",
                    )

                k = model.features.index(feature)
                i = model.gam.terms.feature.index(k)
                XX = model.gam.generate_X_grid(term=i, meshgrid=True)
                if feature == "len":
                    XX = (np.linspace(15, 40, 100),)
                x, pdep = XX[0], model.gam.partial_dependence(i, X=XX, width=0.95, meshgrid=True)

                plt.plot(x, pdep[0], c=color)
                plt.fill_between(x, *pdep[1].T, facecolor=color, alpha=0.2, zorder=-999)

                if model.gam.terms[i].__class__.__name__ == "SplineMultiTerm":
                    multi_spline_axes.append(ax)

            for ax in multi_spline_axes:
                ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
                ax.set_xlim(margined([0, 1]))
                ax.set_ylim(ylim)

        plt.sca(ax_ind)

        for model_name, color, x_offset in zip(model_names, colors, x_offsets):
            model = model_dict[model_name]

            (_, caps, bars) = plt.errorbar(
                0 + x_offset,
                0,
                yerr=[[0], [0]],
                capsize=3,
                marker="o",
                ms=3,
                mfc=color,
                mec=color,
                ecolor=lighten_color(color, 0.6),
            )

            pdep_ind_K = model.gam.partial_dependence(
                model.features.index("ind_K$"), X=(np.array([[1]]),), width=0.95, meshgrid=True
            )

            y = pdep_ind_K[0][0, 0]
            lb = pdep_ind_K[1][0, 0, 0]
            ub = pdep_ind_K[1][0, 0, 1]

            (_, caps, bars) = plt.errorbar(
                1 + x_offset,
                y,
                yerr=[[y - lb], [ub - y]],
                capsize=3,
                marker="o",
                ms=3,
                mfc=color,
                mec=color,
                ecolor=lighten_color(color, 0.6),
            )

        plt.xlim([-0.5, 1.5])
        plt.ylim([ylim[0] / 3.5, ylim[1] / 3.5])
        plt.axhline(0.0, color="k", linestyle="--", linewidth=0.8)

        plt.xticks([0, 1], labels=["R", "K"])

        # TODO: Move to utils
        hydropathy_index = {
            "A": 1.8,
            "R": -4.5,
            "N": -3.5,
            "D": -3.5,
            "C": np.nan,  # 2.5, # since C is Carbamidomethylated
            "Q": -3.5,
            "E": -3.5,
            "G": -0.4,
            "H": -3.2,
            "I": 4.5,
            "L": 3.8,
            "K": -3.9,
            "M": 1.9,
            "F": 2.8,
            "P": -1.6,
            "S": -0.8,
            "T": -0.7,
            "W": -0.9,
            "Y": -1.3,
            "V": 4.2,
        }

        residue_mass = {
            "A": np.float64(71.03711378526002),
            "C": np.float64(160.03064868046997),
            "D": np.float64(115.02694302446001),
            "E": np.float64(129.04259308894),
            "F": np.float64(147.06841391421997),
            "G": np.float64(57.021463720780005),
            "H": np.float64(137.05891185866),
            "I": np.float64(113.0840639787),
            "K": np.float64(128.0949630154),
            "L": np.float64(113.0840639787),
            "M": np.float64(131.04048508865),
            "N": np.float64(114.04292744156),
            "P": np.float64(97.05276384974),
            "Q": np.float64(128.05857750603997),
            "R": np.float64(156.10111102432),
            "S": np.float64(87.03202840486001),
            "T": np.float64(101.04767846934001),
            "V": np.float64(99.06841391422002),
            "W": np.float64(186.07931295091998),
            "Y": np.float64(163.06332853381997),
        }

        gam_splines_dict = {}

        for model_name, color in zip(model_names, colors):
            model = model_dict[model_name]

            gam_splines_dict[model_name] = {}

            for aa in [aa for aa in AA_VOCABULARY if aa not in "RKH"]:
                i = model.gam.terms.feature.index(model.features.index(aa))
                XX = model.gam.generate_X_grid(term=i, meshgrid=True)
                x, pdep = XX[0], model.gam.partial_dependence(i, X=XX, width=0.95, meshgrid=True)

                gam_splines_dict[model_name][aa] = (x, pdep)

        amino_acids = [aa for aa in AA_VOCABULARY if aa not in "RKH"]

        match figure_key:
            case "intermode":
                x_dict = {aa: hydropathy_index[aa] for aa in amino_acids}
            case "intramode":
                x_dict = {aa: residue_mass[aa] for aa in amino_acids}

        y_dict = {
            aa: np.mean(
                [np.mean(gam_splines_dict[model_name][aa][1][0]) for model_name in model_names]
            )
            for aa in amino_acids
        }

        x_arr = np.array([x_dict[aa] for aa in amino_acids])
        y_arr = np.array([y_dict[aa] for aa in amino_acids])

        y_arr = y_arr[~np.isnan(x_arr)]
        x_arr = x_arr[~np.isnan(x_arr)]

        plt.sca(ax_scatter)
        plt.scatter(x_arr, y_arr, s=6, color=SCATTER_COLOR)
        plt.xlim(margined([min(x_dict.values()), max(x_dict.values())], 0.2))
        plt.ylim(margined([min(y_dict.values()), max(y_dict.values())], 0.2))

        match figure_key:
            case "intermode":
                xlabel = "Residue hydropathy index"
                default_angle = 150
                xytext_angle_dict = defaultdict(lambda: default_angle)
                xytext_angle_dict["Q"] = 150
                xytext_angle_dict["E"] = 120
                xytext_angle_dict["N"] = -30
                xytext_angle_dict["D"] = -60
                xytext_angle_dict["Y"] = 150
                xytext_angle_dict["T"] = 120
                xytext_angle_dict["G"] = -30
                xytext_angle_dict["S"] = -60
                xytext_angle_dict["F"] = -30
                xytext_angle_dict["M"] = -60

                xytext_angle_dict["W"] = default_angle + 180
                xytext_angle_dict["C"] = default_angle + 180
                xytext_angle_dict["P"] = default_angle + 180
                xytext_angle_dict["V"] = -60
                xytext_angle_dict["I"] = -30

                xytext_dict = {
                    aa: (
                        1.2 * np.cos(xytext_angle_dict[aa] / 180 * np.pi),
                        1.2 * np.sin(xytext_angle_dict[aa] / 180 * np.pi),
                    )
                    for aa in amino_acids
                }

                r2_text = f"$R^2$ = {np.corrcoef(x_arr, y_arr)[0, 1] ** 2:.2f}"

            case "intramode":
                xlabel = "Residue mass"
                default_angle = 120
                xytext_angle_dict = defaultdict(lambda: default_angle)
                xytext_angle_dict["E"] = 20
                xytext_angle_dict["D"] = -90
                xytext_angle_dict["Q"] = -30
                xytext_angle_dict["N"] = -30
                xytext_angle_dict["P"] = -100
                xytext_angle_dict["S"] = -120
                xytext_angle_dict["T"] = -180
                xytext_angle_dict["C"] = default_angle + 180
                xytext_angle_dict["W"] = default_angle + 180
                xytext_angle_dict["V"] = 150
                xytext_angle_dict["I"] = 160

                xytext_dict = {
                    aa: (
                        1.2 * np.cos(xytext_angle_dict[aa] / 180 * np.pi),
                        1.2 * np.sin(xytext_angle_dict[aa] / 180 * np.pi),
                    )
                    for aa in amino_acids
                }

                r2_text = f"$R^2$ = {np.corrcoef(x_arr, y_arr)[0, 1] ** 2:.2f}"

        plt.annotate(
            r2_text,
            (0, 1),
            xytext=(0.3, -0.4),
            xycoords="axes fraction",
            textcoords="offset fontsize",
            va="top",
            ha="left",
            fontsize="small",
        )

        for aa in amino_acids:
            plt.annotate(
                feature_label_dict[aa],
                (x_dict[aa], y_dict[aa]),
                xytext=xytext_dict[aa],
                ha="center",
                va="center",
                textcoords="offset fontsize",
                arrowprops=dict(
                    arrowstyle="-",
                    facecolor="black",
                    connectionstyle="arc3",
                    patchA=None,
                    shrinkA=5.5,
                    shrinkB=2,
                ),
            )

        plt.xlabel(xlabel)
        print(np.corrcoef(x_arr, y_arr)[0, 1] ** 2)

        plt.show(block=False)


def plot_figure_4de():
    SIMULATION_DATA = load_data.SIMULATION_DATA()
    SIMULATION_PEPTIDES = load_data.SIMULATION_PEPTIDES()

    violin_spacing = 0.19
    violin_horizontal_margin = violin_spacing * 0.925
    between_spacing = 0.7

    violin_max_height = 1.2
    violin_vertical_margin = 0.4

    figsize = (8, 5)
    plt.figure(figsize=figsize)
    simulation_peptides = SIMULATION_PEPTIDES["Gpos_2"]
    axsize = (
        violin_spacing * (len(simulation_peptides) - 1) + 2 * violin_horizontal_margin,
        violin_max_height + 2 * violin_vertical_margin,
    )
    position = compute_axes_position(
        figsize,
        axsize,
        left=1,
        top=1,
    )

    ax_1 = plt.gcf().add_axes(position)
    right = axsize[0] + 1

    simulation_peptides = SIMULATION_PEPTIDES["Gpos_1"]
    axsize = (
        violin_spacing * (len(simulation_peptides) - 1) + 2 * violin_horizontal_margin,
        violin_max_height + 2 * violin_vertical_margin,
    )
    position = compute_axes_position(
        figsize,
        axsize,
        left=right + between_spacing,
        top=1,
    )

    ax_2 = plt.gcf().add_axes(position)

    for ax, data, simulation_peptides, color in [
        (
            ax_1,
            SIMULATION_DATA["Gpos_1_SASA_df"],
            SIMULATION_PEPTIDES["Gpos_1"],
            GAM_SPLINE_UPPER_MODE_COLOR,
        ),
        (
            ax_2,
            SIMULATION_DATA["Gpos_2_SASA_df"],
            SIMULATION_PEPTIDES["Gpos_2"],
            GAM_SPLINE_LOWER_MODE_COLOR,
        ),
    ]:
        plt.sca(ax)

        parts = plt.violinplot(data)
        for partname in ("cbars", "cmins", "cmaxes"):
            parts[partname].set_color("none")
        for pc in parts["bodies"]:
            pc.set_facecolor(color)

        parts = plt.violinplot(data)
        for partname in ("cbars", "cmins", "cmaxes"):
            parts[partname].set_color(color)
            parts[partname].set_linewidth(1)

        for pc in parts["bodies"]:
            pc.set_edgecolor(color)
            pc.set_facecolor("none")
            pc.set_linewidth(1)
            pc.set_alpha(1)

        quartiles_1, medians, quartiles_3 = np.percentile(data.T, [25, 50, 75], axis=1)

        plt.ylabel(r"SASA ($\mathrm{\AA}^2$)")
        plt.xticks(
            range(1, len(simulation_peptides) + 1),
            labels=[peptide.replace("GGG", r"$\bf{GGG}$") for peptide in simulation_peptides],
            va="top",
            ha="right",
            rotation=60,
            rotation_mode="anchor",
        )

        ylim = plt.gca().get_ylim()

        plt.ylim(
            [
                ylim[0] - (ylim[1] - ylim[0]) / violin_max_height * violin_vertical_margin,
                ylim[1] + (ylim[1] - ylim[0]) / violin_max_height * violin_vertical_margin,
            ]
        )

        xlim = [1, len(data.columns)]

        plt.xlim(
            [
                xlim[0] - violin_horizontal_margin / violin_spacing,
                xlim[1] + violin_horizontal_margin / violin_spacing,
            ]
        )

    plt.show(block=False)


def plot_supplementary_figure_s1():
    DATA1 = load_data.DATA1()

    xlabel = "Distance of internal basic site to C-terminus"
    ylabel = "Peptide length"

    xlim = [1 - 0.5, 30 + 0.5]
    ylim = [15 - 0.5, 30 + 0.5]
    min_count = 8
    df = DATA1.query("charge==3 & basic_site_count==3")
    df = df.query("len >= 15")
    df = df.query("not is_acetylated")
    df = df[df["sequence"].apply(lambda x: x[0] in "K")]
    x = df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[2])
    y = df["len"]
    c = df["delta_ccs"] > 0

    idx = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])

    df = df[idx]
    x = x[idx]
    y = y[idx]
    c = c[idx]

    plot_imshow_figure(
        x=x,
        y=y,
        c=c,
        min_count=min_count,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    plt.show(block=False)


def plot_supplementary_figure_s2():
    DATA1 = load_data.DATA1()

    xlabel = "Distance of second internal\nbasic site to C-terminus"
    ylabel = "Distance of first internal\nbasic site to C-terminus"

    xlim = [1 - 0.5, 30 + 0.5]
    ylim = [1 - 0.5, 30 + 0.5]

    min_count = 8

    for tag in ["1_vs_3", "2_vs_3"]:
        df = DATA1.query("charge==3 & basic_site_count==5")
        df = df.query("len >= 15")
        df = df.query("not is_acetylated")
        df = df[df["sequence"].apply(lambda x: x[-1] in "RK")]

        match tag:
            case "1_vs_3":
                x_index = 3
                y_index = 1
                xlabel = "Distance of third internal\nbasic site to C-terminus"
                ylabel = "Distance of first internal\nbasic site to C-terminus"
            case "2_vs_3":
                x_index = 3
                y_index = 2
                xlabel = "Distance of third internal\nbasic site to C-terminus"
                ylabel = "Distance of second internal\nbasic site to C-terminus"

        x = df["len"] - 1 - df["basic_site_positions"].apply(lambda positions: positions[x_index])
        y = df["len"] - 1 - df["basic_site_positions"].apply(lambda positions: positions[y_index])
        c = df["delta_ccs"] > 0

        idx = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])

        df = df[idx]
        x = x[idx]
        y = y[idx]
        c = c[idx]

        fig, axes = plot_imshow_figure(
            x=x,
            y=y,
            c=c,
            min_count=min_count,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
            bin_size=1 / 24,
        )

        plt.show(block=False)


def plot_supplementary_figure_s3():
    DATA1 = load_data.DATA1()

    xlabel = "Distance of internal basic site to C-terminus"
    ylabel = "Peptide length"

    xlim = [1 - 0.5, 30 + 0.5]
    ylim = [15 - 0.5, 30 + 0.5]
    min_count = 3

    for aa in ["R", "K", "H"]:
        df = DATA1.query("charge==3 & basic_site_count==3")
        df = df.query("len >= 15")
        df = df.query("not is_acetylated")
        df = df[df["sequence"].apply(lambda x: x[-1] in "KR")]
        df = df[df["basic_sites"].apply(lambda x: x[1] in aa)]
        x = df["len"] - 1 - df["basic_site_positions"].apply(lambda x: x[1])
        y = df["len"]
        c = df["delta_ccs"] > 0

        idx = (x > xlim[0]) & (x < xlim[1]) & (y > ylim[0]) & (y < ylim[1])

        df = df[idx]
        x = x[idx]
        y = y[idx]
        c = c[idx]

        plot_imshow_figure(
            x=x,
            y=y,
            c=c,
            min_count=min_count,
            xlim=xlim,
            ylim=ylim,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        plt.show(block=False)


def plot_supplementary_figure_s4():
    SIMULATION_DATA = load_data.SIMULATION_DATA()
    SIMULATION_PEPTIDES = load_data.SIMULATION_PEPTIDES()

    for data, simulation_peptides in [
        (
            SIMULATION_DATA["Gpos_1_mean_helix_content_vs_residue"],
            SIMULATION_PEPTIDES["Gpos_1"],
        ),
        (
            SIMULATION_DATA["Gpos_2_mean_helix_content_vs_residue"],
            SIMULATION_PEPTIDES["Gpos_2"],
        ),
    ]:
        figsize = (8, 11)
        plt.figure(figsize=figsize)

        axsize = (1.2, 0.75)
        between_spacing = 0.12
        vertical_between_spacing = 0.24

        n_rows = 3
        n_cols = 5
        N = len(simulation_peptides)

        axes_grid_positions = [
            compute_axes_position(
                figsize,
                axsize,
                left=1 + j * (axsize[0] + between_spacing),
                top=1 + i * (axsize[1] + vertical_between_spacing),
            )
            for i in range(n_rows)
            for j in range(n_cols)
            if i * n_cols + j < N
        ]

        axes_grid = []
        for position in axes_grid_positions:
            ax = plt.gcf().add_axes(position)
            axes_grid.append(ax)

        for i in range(n_rows):
            for j in range(n_cols):
                if i * n_cols + j < N:
                    if j > 0:
                        axes_grid[i * n_cols + j].get_yaxis().set_ticklabels([])
                    if i * n_cols + j + n_cols < N:
                        axes_grid[i * n_cols + j].get_xaxis().set_ticklabels([])

        grid_left = 1
        grid_right = 1 + n_cols * (axsize[0] + between_spacing) - between_spacing
        grid_top = 1
        grid_bottom = (
            1 + n_rows * (axsize[1] + vertical_between_spacing) - vertical_between_spacing
        )

        plt.figtext(
            (grid_left - 25 / PPI) / figsize[0],
            1 - (grid_top + grid_bottom) / 2 / figsize[1],
            r"$\alpha$-Helix fraction",
            ha="right",
            va="center",
            rotation=90,
            fontsize="medium",
        )

        plt.figtext(
            (grid_left + grid_right) / 2 / figsize[0],
            1 - (grid_bottom + 15 / PPI) / figsize[1],
            "Residue position",
            ha="center",
            va="top",
        )

        for i, ax in zip(range(N), axes_grid):
            plt.sca(ax)
            plt.plot(data["residue"], data.iloc[:, i + 1], ".-")
            plt.ylim([0, 1])

            plt.annotate(
                simulation_peptides[i].replace("GGG", r"$\bf{GGG}$"),
                xy=(0.5, 1),
                xytext=(0, 0.3),
                xycoords="axes fraction",
                textcoords="offset fontsize",
                va="bottom",
                ha="center",
                fontsize="small",
            )

        plt.show(block=False)


def plot_supplementary_figure_s5():
    SIMULATION_DATA = load_data.SIMULATION_DATA()
    SIMULATION_PEPTIDES = load_data.SIMULATION_PEPTIDES()

    for key, x_data, y_data, simulation_peptides in [
        (
            "upper_mode",
            SIMULATION_DATA["Gpos_1_SASA_df"],
            SIMULATION_DATA["Gpos_1_helix_content_df"],
            SIMULATION_PEPTIDES["Gpos_1"],
        ),
        (
            "lower_mode",
            SIMULATION_DATA["Gpos_2_SASA_df"],
            SIMULATION_DATA["Gpos_2_helix_content_df"],
            SIMULATION_PEPTIDES["Gpos_2"],
        ),
    ]:
        figsize = (8, 6)
        plt.figure(figsize=figsize)

        axsize = (1.15, 0.75)
        between_spacing = 0.12
        vertical_between_spacing = 0.30

        n_rows = 3
        n_cols = 5
        N = len(simulation_peptides)

        axes_grid_positions = [
            compute_axes_position(
                figsize,
                axsize,
                left=1 + j * (axsize[0] + between_spacing),
                top=1 + i * (axsize[1] + vertical_between_spacing),
            )
            for i in range(n_rows)
            for j in range(n_cols)
            if i * n_cols + j < N
        ]

        axes_grid = []
        for position in axes_grid_positions:
            ax = plt.gcf().add_axes(position)
            axes_grid.append(ax)

        for i in range(n_rows):
            for j in range(n_cols):
                if i * n_cols + j < N:
                    if j > 0:
                        axes_grid[i * n_cols + j].get_yaxis().set_ticklabels([])
                    if i * n_cols + j + n_cols < N:
                        axes_grid[i * n_cols + j].get_xaxis().set_ticklabels([])

        grid_left = 1
        grid_right = 1 + n_cols * (axsize[0] + between_spacing) - between_spacing
        grid_top = 1
        grid_bottom = (
            1 + n_rows * (axsize[1] + vertical_between_spacing) - vertical_between_spacing
        )

        plt.figtext(
            (grid_left - 25 / PPI) / figsize[0],
            1 - (grid_top + grid_bottom) / 2 / figsize[1],
            r"Residue count within an $\alpha$-helix",
            ha="right",
            va="center",
            rotation=90,
            fontsize="medium",
        )

        plt.figtext(
            (grid_left + grid_right) / 2 / figsize[0],
            1 - (grid_bottom + 15 / PPI) / figsize[1],
            r"SASA ($\mathrm{\AA}^2$)",
            ha="center",
            va="top",
        )

        for i, ax in zip(range(N), axes_grid):
            plt.sca(ax)

            plt.scatter(x_data.iloc[:, i], y_data.iloc[:, i], alpha=0.05)

            plt.ylim([0, 12])

            plt.annotate(
                simulation_peptides[i].replace("GGG", r"$\bf{GGG}$"),
                xy=(0.5, 1),
                xytext=(0, 0.3),
                xycoords="axes fraction",
                textcoords="offset fontsize",
                va="bottom",
                ha="center",
                fontsize="small",
            )

        plt.show(block=False)
