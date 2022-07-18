import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 8,
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def set_plot_ratio(ratio):
    # set aspect ratio to 1
    ax = plt.gca()
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

# print("plot_settings loaded _______________")
