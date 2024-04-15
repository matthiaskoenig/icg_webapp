from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import roadrunner
import xarray as xr
from console import console

from settings import icg_model_path
from sampling import samples_for_individual


def load_model(model_path: Path) -> roadrunner.RoadRunner:
    """Load model."""
    r: roadrunner.RoadRunner = roadrunner.RoadRunner(str(model_path))
    r.timeCourseSelections = ["time", "[Cve_icg]"]
    return r


units = {
    "IVDOSE_icg":  'mg',
    "LI__f_oatp1b3":  'dimensionless',
    "BW": 'kg',
    "FVli": 'ml/kg',
    "f_bloodflow":  'dimensionless',
    "f_tissue_loss":  'dimensionless',
    "f_shunts":  'dimensionless',
    "resection_rate":  'dimensionless',
}

def simulate_samples(samples: pd.DataFrame, r: roadrunner.RoadRunner) -> List[pd.DataFrame]:
    """Simulate sample individuals with roadrunner.

    Samples contain multiple samples per individual and possibly multiple
    resection rates.
    """

    changes: Dict = {
        "IVDOSE_icg": np.array(samples['IVDOSE_icg']),  # 'mg',
        "LI__f_oatp1b3": np.array(samples['FOATP1B3']),  # 'dimensionless'),
        "BW": np.array(samples['BMXWT']),  # 'kg'
        "FVli": np.array(samples['LIVVOLKG'])/1000,  # 'ml/kg' -> 'l/kg'
        "f_bloodflow": np.array(samples['f_bloodflow']),  # 'dimensionless'
        "f_tissue_loss": np.array(samples['f_cirrhosis']),  # 'dimensionless'
        "f_shunts": np.array(samples['f_cirrhosis']),  # 'dimensionless'
        "resection_rate": np.array(samples['resection_rate']),  # 'dimensionless'
    }

    # simulate and store in xresult structure
    xres = []

    n_samples = changes["IVDOSE_icg"].size
    # console.print(f"Number of samples: {n_samples}")
    for k in range(n_samples):
        # console.rule(f"sample: {k}", align="left", style="white")
        r.resetAll()
        for key, values in changes.items():
            value = float(values[k])
            # console.print(f"{key} = {value} {units[key]}")
            r.setValue(key, value)
        s = r.simulate(start=0, end=15, steps=100)
        df = pd.DataFrame(s, columns=s.colnames)
        xres.append(df)
    # xres = simulator.run_scan(tc_scan)
    return xres


def calculate_icg_r15(samples_df: pd.DataFrame, dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Calculates pk from results and adds to the model."""

    icg_r15 = np.zeros_like(samples_df["IVDOSE_icg"])
    for k, df in enumerate(dfs):
        # ICG-R15
        #  icg(t=15)/max(icg)  # dimensionless
        icg = df["[Cve_icg]"].values
        icg_t15 = icg[-1]  # last concentration
        # print("icg_t15", icg_t15)
        icg_max = icg.max()
        # print("icg_max", icg_max)
        icg_r15[k] = icg_t15/icg_max
        # print("icg_r15", icg_r15)

    samples_df["postop_r15_model"] = icg_r15
    return samples_df


if __name__ == "__main__":

    samples_df: pd.DataFrame = samples_for_individual(
        bodyweight=75,
        age=55,
        f_cirrhosis=0,
        n=100,
        resection_rates=np.linspace(0.1, 0.9, num=9)
    )
    print(samples_df)
    console.rule(style="white")

    r = load_model(model_path=icg_model_path)
    dfs = simulate_samples(samples_df, r=r)
    samples_df = calculate_icg_r15(samples_df=samples_df, dfs=dfs)

    console.rule(style="white")
    # print(xres)
    print(samples_df.head())
    console.rule(style="white")
