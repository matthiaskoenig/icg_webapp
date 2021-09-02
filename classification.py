"""
- Vary resection rate & create boxplot survival ~ resection rate;
"""
from typing import Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import matplotlib.pyplot as plt

base_path = Path(__file__).parent

font = {
    "weight": "bold",
    "size": 18
}


def fit_classifier():
    """Fits the classification model once."""
    df = pd.read_csv(
        base_path / "data" / "classification_dataset.tsv",
        sep="\t",
        index_col=False
    )

    df.nonsurvival = ~df.nonsurvival.astype(bool)  # nonsurvival=0, survival=1
    X = np.array([df["postop_r15_model"]]).T
    y = np.array(df["nonsurvival"])
    classifier = SVC(kernel='poly', degree=2, probability=True,
                     class_weight='balanced', random_state=42)

    # fit classification model
    classifier.fit(X=X, y=y)
    return classifier


def simulation(samples: pd.DataFrame, resection_rates: np.ndarray) -> pd.DataFrame:
    """Performs model simulations and calculate pharmacokinetic parameters."""

    dfs = []
    for _, rate in enumerate(resection_rates):
        df = samples.copy()
        # simulate

        # calculate pharmacokinetic parameters

        # FIXME replace with simulation
        df["resection_rate"] = rate
        df["postop_pdr_model"] = np.linspace(rate, rate + 0.09, num=len(df))
        df["postop_r15_model"] = np.linspace(rate, rate + 0.09, num=len(df))

        dfs.append(df)

    return pd.concat(dfs)


def classification(classifier, samples: pd.DataFrame) -> pd.DataFrame:
    """Classify samples."""
    predict_X = np.array([samples["postop_r15_model"]]).T
    samples["y_pred"] = classifier.predict(predict_X)
    samples["y_score"] = classifier.predict_proba(predict_X)[:, 1]

    return samples


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
        "y_score": "Survival [-]",
        "postop_r15_model": "Postoperative ICG-R15 [-]",
        "postop_pdr_model": "Postoperative ICG-PDR [-]"
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

    plt.show()
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
    labels = {
        "FOATP1B3": "OATP1B3 [-]",
        "LIVVOLKG": "Liver volume [ml/kg]",
        "LIVBFKG": "Liver blood flow [ml/min/kg]",
    }
    for key, label in labels.items():
        f, ax = plt.subplots(figsize=(6, 6))
        ax.hist(samples[key], bins=15, label="Survivors", color="tab:blue", alpha=1, edgecolor="black", density=True)
        ax.set_xlabel(label, fontdict=font)
        ax.set_ylabel("Density", fontdict=font)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        figures[key] = f

    plt.show()
    return figures


classifier = fit_classifier()

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    from sampling import samples_for_individual

    resection_rates = np.linspace(0.1, 0.9, num=9)
    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        n=15,
    )
    print("-" * 80)
    print(samples.head())

    samples = simulation(samples, resection_rates)
    print("-" * 80)
    print(samples.head())

    samples = classification(
        classifier=classifier,
        samples=samples,
    )
    print("-" * 80)
    print(samples.head())

    # figure boxplots
    # figure_boxplot(samples)
    # figure_histograms(samples)

    import altair as alt

    chart1 = alt.Chart(samples).mark_boxplot(size=35).encode(
        x='resection_rate:Q',
        y='y_score:Q'
    ).configure_mark(
        fill="white",
        stroke="black"
    )

    # update this:
    # https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html?highlight=boxplot#altair.Chart.mark_boxplot

    chart1.show()
