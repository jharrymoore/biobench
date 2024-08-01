import pytest
import biobench.utils
from openmm.utils import get_available_platforms


def test_density_experiment():
    if "cuda" not in get_available_platforms():
        pytest.skip("CUDA device not available.")
    pass


def test_slurm_submission():
    if not biobench.utils.which("sbatch"):
        pytest.skip("Slurm not available.")

