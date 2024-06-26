"""
- Vary resection rate & create boxplot survival ~ resection rate;
"""
from typing import Dict

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC

from sampling import samples_for_individual
from settings import icg_model_path
from simulation import simulate_samples, calculate_icg_r15, load_model

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


classifier = fit_classifier()


def classification(samples: pd.DataFrame) -> pd.DataFrame:
    """Classify samples."""
    predict_X = np.array([samples["postop_r15_model"]]).T
    samples["y_pred"] = classifier.predict(predict_X).astype(int)
    samples["y_score"] = classifier.predict_proba(predict_X)[:, 1]

    return samples


def example_classification(f_cirrhosis=0) -> pd.DataFrame():
    """Example classification for testing."""
    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        f_cirrhosis=f_cirrhosis,
        n=100,
        resection_rates=np.linspace(0.1, 0.9, num=9)
    )

    simulator = load_model(model_path=icg_model_path)
    dfs = simulate_samples(samples, simulator)
    samples: Dict = calculate_icg_r15(samples_df=samples, dfs=dfs)
    samples_df: pd.DataFrame = classification(samples=samples)
    return samples_df


if __name__ == "__main__":

    samples = example_classification()
    print("-" * 80)
    print(samples.head())
    samples.t

