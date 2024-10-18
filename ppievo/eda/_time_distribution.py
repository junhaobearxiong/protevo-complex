import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from ppievo.io import read_transitions
from ppievo.utils import get_quantization_points_from_geometric_grid, get_quantile_idx

PLT_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
}


def plot_time_distribution(
    transitions_dir: str,
    pair_names: Optional[list[str]] = None,
    quantiles: list[Union[str, float]] = get_quantization_points_from_geometric_grid(),
    title: Optional[str] = "Time distribution",
    output_dir: Optional[str] = None,
    ax=None,
):
    quantiles = np.array(sorted([float(x) for x in quantiles]))
    counts = np.zeros((len(quantiles), len(quantiles)))
    if pair_names is None:
        pair_names = [
            file.split(".txt")[0]
            for file in os.listdir(transitions_dir)
            if file.endswith(".txt")
        ]
    for pair_name in pair_names:
        transitions = read_transitions(
            os.path.join(transitions_dir, pair_name + ".txt")
        )
        for x1, y1, x2, y2, tx, ty in transitions:
            # Get the quantization index for the distance of x and y separately
            qidx_x = get_quantile_idx(quantiles=quantiles, t=tx)
            qidx_y = get_quantile_idx(quantiles=quantiles, t=ty)
            counts[qidx_x, qidx_y] += 1

    # Helper function to plot bivariate histogram
    def plot_heatmap(ax, data, x_label, y_label, title):
        quantiles_labels = [f"{q:.1e}" for q in quantiles]
        df = pd.DataFrame(data, index=quantiles_labels, columns=quantiles_labels)
        sns.heatmap(
            data=df.iloc[::-1],
            cmap="YlGnBu",
            cbar=True,
            xticklabels=10,
            yticklabels=10,
            ax=ax,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    def plot_histplot(ax, data):
        # Reshape the data for sns.histplot
        y_indices, x_indices = np.meshgrid(
            np.arange(len(quantiles)), np.arange(len(quantiles))
        )
        data = pd.DataFrame(
            {
                "Time X (bucket)": quantiles[x_indices.flatten()],
                "Time Y (bucket)": quantiles[y_indices.flatten()],
                "Count": counts.flatten(),
            }
        )
        data = data[data["Count"] > 0]
        # Create the bivariate histogram using seaborn
        sns.histplot(
            data=data,
            x="Time X (bucket)",
            y="Time Y (bucket)",
            weights="Count",
            cbar=True,
            bins=len(quantiles),
            ax=ax,
        )

    # Create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    plot_heatmap(
        ax=ax,
        data=counts,
        x_label="Time Y (bucket)",
        y_label="Time X (bucket)",
        title=title,
    )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "time_distribution.png"), **PLT_SAVEFIG_KWARGS
        )
        plt.close()
    return counts
