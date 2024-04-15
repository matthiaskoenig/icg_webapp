"""Tests for the web app."""

import numpy as np
import pandas as pd

from sampling import samples_for_individual
from simulation import simulate_samples, calculate_icg_r15, load_model
from settings import icg_model_path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def test_sampling() -> None:
    """Test creation of the individual samples."""

    samples: pd.DataFrame = samples_for_individual(
        bodyweight=50,
        age=80,
        f_cirrhosis=0.1,
        n=15,
        resection_rates=None,
    )

    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 15
    assert samples.BMXWT.unique()[0] == 50
    assert samples.f_cirrhosis.unique()[0] == 0.1
    assert samples.AGE.unique()[0] == 80.0


def test_sampling_resection_rate() -> None:
    """Test creation of the individual samples with resection rates."""

    samples: pd.DataFrame = samples_for_individual(
        bodyweight=75,
        age=55,
        f_cirrhosis=0,
        n=15,
        resection_rates=np.linspace(0, 1, num=10),
    )

    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 15 * 10
    assert samples.BMXWT.unique()[0] == 75
    assert samples.AGE.unique()[0] == 55
    assert samples.f_cirrhosis.unique()[0] == 0
    assert len(samples.resection_rate.unique()) == 10


def test_simulate() -> None:
    """Test simlation and PK calculation"""
    samples = samples_for_individual(
        bodyweight=75,
        age=55,
        f_cirrhosis=0,
        n=5,
        resection_rates=np.linspace(0.1, 0.9, num=9)
    )

    simulator = load_model(model_path=icg_model_path)
    xres, samples = simulate_samples(simulator=simulator, samples=samples)
    assert isinstance(samples, pd.DataFrame)

    samples = calculate_icg_r15(samples_df=samples, xres=xres)
    assert isinstance(samples, pd.DataFrame)
    assert "postop_r15_model" in samples.columns
