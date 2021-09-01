"""
- Copy trainings data here `classificaiton_data.tsv
- Only fit the optimal classifier for PBPK1
- Use the classifier to predict the data
    # predict test data

    y_pred = classifier.predict(X)
    y_score = classifier.predict_proba(X)[:, 1]

- Vary resection rate & create boxplot survival ~ resection rate;
"""
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    from sampling import samples_for_individual

    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        n=15,
    )

    samples["postop_r15_model"] = np.linspace(0, 1, num=len(samples)) # dummy normal distribution
    print(samples)
