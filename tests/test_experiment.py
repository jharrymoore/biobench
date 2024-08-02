import pytest
import biobench.utils
from openmmtools.utils import get_available_platforms
from biobench.config import Configuration, SimulationParams, ValidationDataPaths, ExecutionEnvironment
from biobench.experiment import DensityExperiment
import os
import torch
import mpiplus

@pytest.mark.parametrize("optimized_model", [True])
@pytest.mark.parametrize("minimiser", [None, "openmm", "ase"])
def test_density_experiment(optimized_model: bool, minimiser: str):
    platforms = [p.getName() for p in get_available_platforms()]
    if "CUDA" not in platforms:
        pytest.skip("CUDA device not available.")
    mpiplus.get_mpicomm()
    model_path = os.path.join(os.path.dirname(__file__), "test_data/maceoff_sc.model")
    simulation_params = SimulationParams(
        steps=100,
        pressure=1.0,
        minimiser=minimiser,
        optimized_model=optimized_model,
        temp=298
    )
    experiment = DensityExperiment(
        data_path=ValidationDataPaths.density.value,
        model_path=model_path,
        execution_environment=ExecutionEnvironment.LOCAL,
        overwrite=False,
        steps=1000,
        simulation_params=simulation_params
        run_max_n=4
    )
    result = experiment.execute()

    output = experiment.analyse()
    

def test_slurm_submission():
    if not biobench.utils.which("sbatch"):
        pytest.skip("Slurm not available.")

