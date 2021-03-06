"""Module for sampling in silico individuals."""

from typing import Dict, Tuple
import logging

import pandas as pd
import numpy as np
from pathlib import Path
import libsbml
import json
from scipy.stats import lognorm, norm

from settings import icg_model_path

logger = logging.getLogger(__name__)
base_path = Path(__file__).parent


# read parameters from SBML
doc: libsbml.SBMLDocument = libsbml.readSBMLFromFile(str(icg_model_path))
model: libsbml.Model = doc.getModel()
COBW = model.getParameter("COBW").getValue()
FVli = model.getParameter("FVli").getValue()

# load distributions from JSON
with open(base_path / "data" / "oatp1b3_distribution.json", "r") as f_json:
    oatp1b3_distribution: Tuple = json.load(f_json)

with open(base_path / "data" / "cov_liver_volume_bloodflow.json", "r") as f_json:
    cov_liver_volume_bloodflow: Dict = json.load(f_json)
    # fix covariance matrix
    for _, info in cov_liver_volume_bloodflow.items():
        info['cov'] = np.array(info['cov'])


def sample_liver_volume_bloodflow(samples: pd.DataFrame, liver_volume: float = np.NaN,
                                  liver_bloodflow: float = np.NaN) -> pd.DataFrame:
    """Adds liver volume and blood flow information to samples.

    Changes samples in place.
    FIXME: extend to support 1D sampling if either volume of blood flow is given
    """
    # read covariance information
    cov_info = cov_liver_volume_bloodflow

    livvolkg_values = []
    livbfkg_values = []

    for _, row in samples.iterrows():
        group = row["AGECLASS"]
        dist_info = cov_info[group]


        vol_bf = np.random.multivariate_normal(
            (dist_info["volbw_mean"], dist_info["bfbw_mean"]),
            cov=dist_info["cov"],
            size=1
        )
        livvolkg = vol_bf[0][0]
        livbfkg = vol_bf[0][1]

        # FIXME: necessary to use the conditional normal distributions
        # see: https://stackoverflow.com/questions/38713746/python-numpy-conditional-simulation-from-a-multivatiate-distribution
        # see https://en.wikipedia.org/wiki/Multivariate_normal_distribution: conditional distributions
        if liver_volume and not np.isnan(liver_volume):
            # assume small variation around provided value (due to measurement variation)
            livvolkg = norm.rvs(
                 loc=liver_volume / row["BMXWT"],
                 scale=0.05 * liver_volume / row["BMXWT"],
            )

        if liver_bloodflow and not np.isnan(liver_bloodflow):
            # assume small variation around provided value (due to measurement variation
            livbfkg = norm.rvs(
                loc=liver_bloodflow / row["BMXWT"],
                scale=0.05 * liver_bloodflow / row["BMXWT"],
            )

        livvolkg_values.append(livvolkg)
        livbfkg_values.append(livbfkg)

    samples["LIVVOLKG"] = livvolkg_values
    samples["LIVBFKG"] = livbfkg_values

    return samples


def samples_for_individual(
        bodyweight: float,
        age: float,
        liver_volume: float = np.NaN,
        liver_bloodflow: float = np.NaN,
        f_cirrhosis: float = 0.0,
        n: int = 100,
        resection_rates: np.ndarray = None,
        random_seed: int = None,
):
    """Sample data for given individual."""
    if random_seed:
        np.random.seed(random_seed)

    if age <= 40:
        ageclass = "18-40"
    elif 40 < age <= 65:
        ageclass = "41-65"
    elif age > 65:
        ageclass = "66-84"

    # empty data frame with n copies:
    samples = pd.DataFrame({
        "AGE": [age] * n,
        "AGECLASS": [ageclass] * n,
        "BMXWT": [bodyweight] * n,
        "f_cirrhosis": [f_cirrhosis] * n,
    })

    # oatp
    oatp_pars = oatp1b3_distribution
    oatp_dist = lognorm(s=oatp_pars['s'], scale=oatp_pars['scale'])
    oatp_samples = oatp_dist.rvs(size=len(samples))
    samples["FOATP1B3"] = oatp_samples

    # dose
    samples["IVDOSE_icg"] = 0.5 * samples["BMXWT"]

    # liver volume and hbf
    samples = sample_liver_volume_bloodflow(
        samples=samples,
        liver_volume=liver_volume,
        liver_bloodflow=liver_bloodflow,
    )
    samples["f_bloodflow"] = samples["LIVBFKG"]/(COBW * 60.0/1000.0 * FVli/1000)

    samples["LIVVOL"] = samples["LIVVOLKG"] * bodyweight
    samples["LIVBF"] = samples["LIVBFKG"] * bodyweight

    # include resection rates in samples, i.e. in
    # all individudals different resections are performed
    if resection_rates is not None:
        dfs = []
        for rate in resection_rates:
            df = samples.copy()
            df['resection_rate'] = rate
            dfs.append(df)

        samples = pd.concat(dfs)

    return samples
