"""
- Vary resection rate & create boxplot survival ~ resection rate;
"""
from typing import Tuple

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

base_path = Path(__file__).parent


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


def figure(predictions):
    """Boxplots for predictions.

    FIXME:
    - histogram liver volume, liver blood flow, oatp1b3
    - create boxplots of postop R15 ~ resection_rate
    - create boxplots of postop PDR ~ resection_rate
    - pimp the plots: larger fonts, bold, labels; show outliers (make sure)

    # make interactive plot
    - use altair
    - (plotly)

    """

    resection_rates = sorted(predictions.resection_rate.unique())
    n_rates = len(resection_rates)

    f = plt.figure(figsize=(n_rates, 6))
    gs = GridSpec(1, n_rates)
    gs.update(wspace=0, hspace=0)

    for k, rate in enumerate(resection_rates):
        df = predictions[predictions.resection_rate == rate]
        ax = f.add_subplot(gs[0:1, k:k + 1])
        _ = boxplot(ax, df.y_score, k, rate, n_rates)

    plt.show()
    return f


def boxplot(ax, data, k, rate, n_rates):
    ax.tick_params(axis="x", labelsize=9)
    if k != (n_rates-1):
        ax.spines['right'].set_visible(False)
    if k != 0:
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
    if k == 0:
        ax.set_ylabel("Survival [-]")

    ax.set_ylim(top=1, bottom=0)

    ax.boxplot(data, labels=[np.round(rate, decimals=1)], widths=0.9, showmeans=True)
    return ax


classifier = fit_classifier()

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    from sampling import samples_for_individual

    resection_rates = np.linspace(0.1, 0.9, num=9)
    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        f_cirrhosis=0,
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
    figure(samples)


    import altair as alt

    chart1 = alt.Chart(samples).mark_boxplot().encode(
        x='resection_rate:Q',
        y='y_score:Q'
    )
    chart2 = alt.Chart(samples).mark_line().encode(
        x='resection_rate:Q',
        y='y_score:Q'
    )
    # update this:
    # https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html?highlight=boxplot#altair.Chart.mark_boxplot
    (chart1 + chart2).show()
