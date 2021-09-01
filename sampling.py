"""Module for sampling in silico individuals."""

from typing import Dict, Tuple
import logging

import pandas as pd
import numpy as np
from pathlib import Path
import libsbml
import json


logger = logging.getLogger(__name__)
base_path = Path(__file__).parent

# read parameters from SBML
model_path = str(base_path / "model" / "icg_body_flat.xml")
doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(model_path)
model: libsbml.Model = doc.getModel()
COBW = model.getParameter("COBW").getValue()
FVli = model.getParameter("FVli").getValue()

# load distributions from JSON
with open(base_path / "data" / "oatp1b3_distribution.json", "r") as f_json:
    oatp1b3_distribution: Tuple = json.load(f_json)

with open(base_path / "data" / "cov_liver_volume_bloodflow.json", "r") as f_json:
    cov_liver_volume_bloodflow: Dict = json.load(f_json)


def sample_liver_volume_bloodflow(samples: pd.DataFrame) -> pd.DataFrame:
    """Adds liver volume and blood flor information to samples.

    Changes samples in place.
    FIXME: extend to support 1D sampling if either volume of blood flow is given
    """
    # read covariance information
    cov_info = cov_liver_volume_bloodflow

    livvolkg = []
    livbfkg = []

    for _, row in samples.iterrows():
        group = row["AGECLASS"]
        dist_info = cov_info[group]

        vol_bf = np.random.multivariate_normal(
            (dist_info["volbw_mean"], dist_info["bfbw_mean"]),
            cov=dist_info["cov"],
            size=1
        )
        livvolkg.append(vol_bf[0][0])
        livbfkg.append(vol_bf[0][1])

    samples["LIVVOLKG"] = livvolkg
    samples["LIVBFKG"] = livbfkg

    return samples


def samples_for_individual(bodyweight: float, age: float, n: int = 100, random_seed: int = 42):
    """Sample data for given individual."""
    np.random.seed(random_seed)

    if age <= 40:
        ageclass = "18-40"
    elif 40 < age <=65:
        ageclass = "41-65"
    elif age > 65:
        ageclass = "66-84"

    # empty data frame with n copies:
    samples = pd.DataFrame({
        "AGE": [age] * n,
        "AGECLASS": [ageclass] * n,
        "BMXWT": [bodyweight] * n,
    })
    print(samples)

    # oatp
    oatp_dist, oatp_unit, oatp_pars = oatp1b3_distribution
    oatp_samples = oatp_dist.rvs(size=len(samples))
    samples["FOATP1B3"] = oatp_samples

    # bsa via formula
    samples["BSA"] = 0.024265 * np.power(samples["BMXWT"], 0.5378) * np.power(
        samples["BMXHT"], 0.3964)

    # dose
    samples["IVDOSE_icg"] = 0.5 * samples["BMXWT"]

    # liver volume and hbf
    samples = sample_liver_volume_bloodflow(samples=samples)
    samples["f_bloodflow"] = samples["LIVBFKG"]/(COBW * 60.0/1000.0 * FVli/ 1000)

    return samples


if __name__ == "__main__":
    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        n=15,
    )