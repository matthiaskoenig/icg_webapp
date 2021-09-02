from pathlib import Path
from typing import Dict

import pandas as pd

from sampling import samples_for_individual
import numpy as np
import xarray as xr

from sbmlsim.utils import timeit

from sbmlsim.simulation import Timecourse, TimecourseSim, ScanSim, Dimension
# from sbmlsim.simulator import SimulatorSerial as Simulator
from sbmlsim.simulator.simulation_ray import SimulatorParallel as Simulator

icg_model_path = Path(__file__).parent / "model" / "icg_body_flat.xml"


def load_model(model_path: Path) -> Simulator:
    """Load model."""
    simulator = Simulator(model_path)
    simulator.set_timecourse_selections(["time", "[Cve_icg]"])
    return simulator

simulator = load_model(icg_model_path)


@timeit
def simulate_samples(samples, resection_rates: np.ndarray) -> xr.Dataset:
    """Simulate model."""
    # include resection rates in samples
    dfs = []
    for rate in resection_rates:
        df = samples.copy()
        df['resection_rate'] = rate
        dfs.append(df)

    samples = pd.concat(dfs)

    Q_ = simulator.Q_
    tc = Timecourse(
        start=0, end=15, steps=100,
        # start=0, end=15, steps=1,
        changes={}
    )
    changes = {
        "LI__f_oatp1b3": Q_(np.array(samples['FOATP1B3']), 'dimensionless'),
        "BW": Q_(np.array(samples['BMXWT']), 'kg'),
        "FVli": Q_(np.array(samples['LIVVOLKG']), 'ml/kg'),
        "f_bloodflow": Q_(np.array(samples['f_bloodflow']), 'dimensionless'),
        "IVDOSE_icg": Q_(np.array(samples['IVDOSE_icg']), 'mg'),

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
    print("icg_t15", icg_t15)
    icg_max = xres["[Cve_icg]"].max(dim="_time")
    print("icg_max", icg_max)
    icg_r15 = icg_t15/icg_max
    print("icg_r15", icg_r15)
    samples["postop_r15_model"] = icg_r15.values

    return samples

if __name__ == "__main__":

    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        f_cirrhosis=0,
        n=100,
    )
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(samples.head())

    resection_rates = np.linspace(0.1, 0.9, num=9)
    xres, samples = simulate_samples(samples, simulator, resection_rates)

    print("-" * 80)
    print(xres)
    samples = calculate_pk(samples=samples, xres=xres)
    print(samples.head())





