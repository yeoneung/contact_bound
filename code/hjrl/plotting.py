"""Matplotlib defaults: fixed categorical order, thin marks, recessive axes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.6, "font.size": 9, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=C), "figure.dpi": 140, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    return plt
