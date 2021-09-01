"""
- Vary resection rate & create boxplot survival ~ resection rate;
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC

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
    print(y_pred)
    print(y_score)


if __name__ == "__main__":
    from sampling import samples_for_individual

    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        n=15,
    )

    samples["postop_r15_model"] = np.linspace(0, 1, num=len(samples))  # dummy normal distribution
    print(samples)

    classification(samples=samples)
