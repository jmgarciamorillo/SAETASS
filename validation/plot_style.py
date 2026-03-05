import matplotlib.pyplot as plt


def apply_plot_style():
    """
    Apply unified plot style settings for validation cases.
    """
    # Use LaTeX for matplotlib text rendering when available
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
            # increased font sizes for publication-quality figures
            "font.size": 12,
            "axes.titlesize": 25,
            "axes.labelsize": 25,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "lines.linewidth": 3,
            # Automatically adjust subplot parameters to give specified padding
            # and prevent axes labels from being cut off
            "figure.autolayout": True,
            "savefig.bbox": "tight",
        }
    )
