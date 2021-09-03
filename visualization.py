"""
- Vary resection rate & create boxplot survival ~ resection rate;
"""
from typing import Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import matplotlib.pyplot as plt

from classification import example_classification

base_path = Path(__file__).parent

font = {
    "weight": "bold",
    "size": 18
}


def figure_boxplot(predictions):
    """Boxplots for predictions.

    # make interactive plot
    - use altair
    - (plotly)

    """

    resection_rates = sorted(predictions.resection_rate.unique())
    n_rates = len(resection_rates)

    gs = GridSpec(1, n_rates)
    gs.update(wspace=0, hspace=0)

    labels = {
        "y_score": "Survival probability [-]",
        # "y_pred": "Survival [-]",
        "postop_r15_model": "Postoperative ICG-R15 [-]",
        # "postop_pdr_model": "Postoperative ICG-PDR [-]"
    }

    figures = {}
    for key, ylabel in labels.items():
        f = plt.figure(figsize=(n_rates, 6))
        f.suptitle('Resection rate [-]', y=0.05, size=17, weight="bold")
        for k, rate in enumerate(resection_rates):
            df = predictions[predictions.resection_rate == rate]
            ax = f.add_subplot(gs[0:1, k:k + 1])
            _ = boxplot(ax, df[key], k, rate, n_rates, ylabel)
        figures[key] = f

    return figures


def boxplot(ax, data, k, rate, n_rates, ylabel):
    ax.tick_params(axis="x", labelsize=17)
    ax.tick_params(axis="y", labelsize=17)
    if k != (n_rates-1):
        ax.spines['right'].set_visible(False)
    if k != 0:
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
    if k == 0:
        ax.set_ylabel(ylabel, fontdict=font)

    ax.set_ylim(top=1, bottom=0)

    ax.boxplot(data, labels=[np.round(rate, decimals=1)], widths=0.9, showmeans=True, showfliers=True)
    return ax


def figure_histograms(samples):
    figures = {}
    infos = {
        "FOATP1B3": {
            "label": "OATP1B3 [-]",
            "color": "tab:blue",
            "xlim": [0, 6]
        },
        # "LIVVOLKG": "Liver volume [ml/kg]",
        # "LIVBFKG": "Liver blood flow [ml/min/kg]",
        "LIVVOL": {
            "label": "Liver volume [ml]",
            "color": "tab:orange",
            "xlim": [500, 3000]
        },
        "LIVBF": {
            "label": "Hepatic blood flow [ml/min]",
            "color": "tab:red",
            "xlim": [500, 3000]
        },
    }
    for k, key in enumerate(infos):
        info = infos[key]
        f, ax = plt.subplots(figsize=(6, 6))

        ax.hist(samples[key], bins=15, label="Survivors", color=info["color"],
                alpha=0.9, edgecolor="black", density=False)
        ax.set_xlabel(info["label"], fontdict=font)
        ax.set_ylabel("Frequency", fontdict=font)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        if info["xlim"]:
            ax.set_xlim(info["xlim"])
        figures[key] = f

    return figures


if __name__ == "__main__":
    samples = example_classification()
    print("-" * 80)
    print(samples.head())

    # figure boxplots
    figure_boxplot(samples)
    figure_histograms(samples)

    plt.show()

    import altair as alt

    chart1 = alt.Chart(samples).mark_boxplot(size=35).encode(
        x=alt.X('resection_rate:Q', axis=alt.Axis(format='%', title='Resection rate')),
        y=alt.Y('y_score:Q', axis=alt.Axis(format='%', title='Survival'))
    )

    chart2 = alt.Chart(samples).mark_tick(color="red", width=35).encode(
        x='resection_rate:Q',
        y='mean(y_score)',
        tooltip="mean(y_score)",
    ).interactive()

    charts = alt.layer(chart1, chart2).properties(
        title="Survival probabilities"
    ).configure_mark(fill="white", stroke="black")

    # update this:
    # https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html?highlight=boxplot#altair.Chart.mark_boxplot

    # (chart1 + chart2).show()
    charts.show()
