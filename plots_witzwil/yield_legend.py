#%%
import matplotlib.pyplot as plt
import os
import sys

while "code/plots_witzwil" in os.getcwd():
    os.chdir("..")
sys.path.append(os.getcwd())
import my_utils.plot_settings

# colors = [
# 	"#000000",
# 	"#222222",
# 	"#444444",
# 	"#666666",
# 	"#888888",
# 	"#aaaaaa",
# 	"#cccccc",
# 	"#eeeeee"
# ]

colors = [
	"#000000ff",
	"#000000cc",
	"#000000aa",
	"#00000088",
	"#00000066",
	"#00000044",
	"#00000022",
	"#00000000"
]
f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(8)]
labels = ["0.1",
"0.85",
"1.6",
"2.35",
"3.1",
"3.85",
"4.6",
"5.35"]
legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=False, title="yield in [t/ha]")

def export_legend(legend, filename="../latex/figures/misc/yield_legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
# plt.show()

