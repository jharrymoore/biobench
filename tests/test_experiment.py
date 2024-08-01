import pytest
import biobench.utils
from openmmtools.utils import get_available_platforms
from biobench.config import Configuration, SimulationParams, ValidationDataPaths, ExecutionEnvironment
from biobench.experiment import DensityExperiment
import os
import torch
import mpiplus


def test_density_experiment():
    if "cuda" not in get_available_platforms():
        pytest.skip("CUDA device not available.")
    mpiplus.get_mpicomm()
    model_path = os.path.join(os.path.dirname(__file__), "test_data/maceoff_sc.model)")
    model = torch.load(model_path)
    simulation_params = SimulationParams(
        steps=1000,
        pressure=1.0,
    )
    experiment = DensityExperiment(
        data_path=ValidationDataPaths.density.value,
        model=model,
        execution_environment=ExecutionEnvironment.LOCAL,
        overwrite=False,
        steps=1000,
        simulation_params=simulation_params
    )
    result = experiment.execute()
    output = experiment.analyse()
    

def test_slurm_submission():
    if not biobench.utils.which("sbatch"):
        pytest.skip("Slurm not available.")

