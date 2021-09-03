from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import xarray as xr

from sbmlsim.utils import timeit
from sbmlsim.simulation import Timecourse, TimecourseSim, ScanSim, Dimension
from sbmlsim.simulator import SimulatorSerial as Simulator
# from sbmlsim.simulator.simulation_ray import SimulatorParallel as Simulator

from settings import icg_model_path
from sampling import samples_for_individual


def load_model(model_path: Path) -> Simulator:
    """Load model."""
    simulator = Simulator(
        model_path,
        absolute_tolerance=1e-8,
        relative_tolerance=1e-8,
    )
    simulator.set_timecourse_selections(["time", "[Cve_icg]"])
    return simulator


@timeit
def simulate_samples(samples: pd.DataFrame, simulator) -> Tuple[xr.Dataset, pd.DataFrame]:
    """Simulate sample individuals with given simulator.

    Samples contain multiple samples per individual and possibly multiple
    resection rates.
    """
    Q_ = simulator.Q_
    tc = Timecourse(
        start=0, end=15, steps=100,
        changes={}
    )
    changes: Dict = {
        "IVDOSE_icg": Q_(np.array(samples['IVDOSE_icg']), 'mg'),
        "LI__f_oatp1b3": Q_(np.array(samples['FOATP1B3']), 'dimensionless'),
        "BW": Q_(np.array(samples['BMXWT']), 'kg'),
        "FVli": Q_(np.array(samples['LIVVOLKG']), 'ml/kg'),
        "f_bloodflow": Q_(np.array(samples['f_bloodflow']), 'dimensionless'),
        "f_tissue_loss": Q_(np.array(samples['f_cirrhosis']), 'dimensionless'),
        "f_shunts": Q_(np.array(samples['f_cirrhosis']), 'dimensionless'),
        "resection_rate": Q_(np.array(samples['resection_rate']), 'dimensionless'),
    }
    tc_scan = ScanSim(
        simulation=TimecourseSim([tc]),
        dimensions=[
            Dimension("samples", changes=changes),
        ]
    )
    xres = simulator.run_scan(tc_scan)
    return xres, samples


@timeit
def calculate_pk(samples, xres) -> pd.DataFrame:
    """Calculates pk from results and adds to the model."""

    # ICG-R15
    #  icg(t=15)/max(icg)  # dimensionless
    icg_t15 = xres.isel(_time=-1)["[Cve_icg]"]
    # print("icg_t15", icg_t15)
    icg_max = xres["[Cve_icg]"].max(dim="_time")
    # print("icg_max", icg_max)
    icg_r15 = icg_t15/icg_max
    # print("icg_r15", icg_r15)
    samples["postop_r15_model"] = icg_r15.values

    return samples


if __name__ == "__main__":

    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        f_cirrhosis=0,
        n=100,
        resection_rates=np.linspace(0.1, 0.9, num=9)
    )
    simulator = load_model(model_path=icg_model_path)

    xres, samples = simulate_samples(samples, simulator=simulator)
    samples = calculate_pk(samples=samples, xres=xres)

    print("-" * 80)
    print(xres)
    print(samples.head())
    print("-" * 80)
