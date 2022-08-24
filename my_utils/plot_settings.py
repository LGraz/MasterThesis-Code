import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 8,
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def set_plot_ratio(ratio):
    # set aspect ratio to 1
    ax = plt.gca()
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)


def legend_handle_obj(label, color):
    return Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=label,
        markerfacecolor=color,
        markersize=8,
    )


def legend_scl45_grey(**kwargs):
    plt.legend(
        handles=[
            legend_handle_obj("SCL45", "#000000"),
            legend_handle_obj("non SCL45", "#cccccc"),
        ],
        **kwargs
    )


def legend_scl(**kwargs):
    plt.legend(
        handles=[
            legend_handle_obj("2: Dark features", "#404040"),
            legend_handle_obj("3: Shadows", "#bf8144"),
            legend_handle_obj("4: Vegetation", "#00ff3c"),
            legend_handle_obj("5: Soils", "#ffed50"),
            legend_handle_obj("6: Water", "#0d00fa"),
            legend_handle_obj("7: Cloud -", "#808080"),
            legend_handle_obj("8: Cloud", "#bfbfbf"),
            legend_handle_obj("9: Cloud +", "#eeeeee"),
            legend_handle_obj("10: Cirrus", "#0bb8f0"),
            legend_handle_obj("11: Snow", "#ffbfbf"),
        ],
        **kwargs
    )


# print("plot_settings loaded _______________")
