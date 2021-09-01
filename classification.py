"""
- Vary resection rate & create boxplot survival ~ resection rate;
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

base_path = Path(__file__).parent

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def classification(samples: pd.DataFrame):
    df = pd.read_csv(
        base_path / "data" / "classification_dataset.tsv",
        sep="\t",
        index_col=False
    )

    df.nonsurvival = df.nonsurvival.astype(bool)
    X = np.array([df["postop_r15_model"]]).T
    y = np.array(df["nonsurvival"])
    classifier = SVC(kernel='poly', degree=2, probability=True,
                     class_weight='balanced', random_state=42)

    # fit classification model
    classifier.fit(X=X, y=y)

    # predict data
    predict_X = np.array([samples["postop_r15_model"]]).T
    y_pred = classifier.predict(predict_X)
    y_score = classifier.predict_proba(predict_X)[:, 1]
    # print(y_pred)
    # print(y_score)
    return y_score


def figure(predictions):
    f = plt.figure(figsize=(len(predictions), 6))
    gs = GridSpec(1, len(predictions))
    gs.update(wspace=0, hspace=0)
    for k, prediction in enumerate(predictions):
        ax = f.add_subplot(gs[0:1, k:k + 1])
        ax = boxplot(ax, prediction, k)

    plt.show()
    return f


def boxplot(ax, data, k):
    ax.tick_params(axis="x", labelsize=9)
    if k != 9:
        ax.spines['right'].set_visible(False)
    if k != 0:
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
    if k == 0:
        ax.set_ylabel("Nonsurvival [-]")
    ax.set_ylim(top=1, bottom=0)

    ax.boxplot(data, labels=[k], widths=0.9, showmeans=True)
    return ax


if __name__ == "__main__":
    from sampling import samples_for_individual

    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        n=15,
    )

    predictions = []
    # iterate over resection rates
    for k in np.linspace(0, 0.9, num=10):
        samples["postop_r15_model"] = np.linspace(k, k+0.09, num=len(samples))  # dummy icg_r15
        y_score = classification(samples=samples)
        predictions.append(y_score)

    # figure boxplots
    figure(predictions)
