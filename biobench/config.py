from dataclasses import dataclass
from enum import Enum

class ValidationDataPaths(Enum):
    density = "https://drive.google.com/uc?id=18RurJW7n7dWYx3vKSZcuN9ejf_FuwdPo"
    solvation = None

@dataclass
class Configuration:
    data_path: str
    output_dir: str
    partition: str
    timelimit: str
    account: str
    overwrite: bool
    model_path: str
    steps: int

class ExecutionEnvironment(Enum):
    SLURM = 1
    LOCAL = 2


@dataclass
class ExperimentResult:
    system_name: str
    result: float
    experimental: float


@dataclass
class SlurmParams:
    partition: str
    timelimit: str
    account: str
    overwrite: bool = False
    tasks: int = 1
    gpus: int  = 1
    nodes: int = 1


@dataclass
class SimulationParams:
    steps: int
    pressure: float
    nl: str
    minimiser: str
    temp: float
    optimized_model: bool

