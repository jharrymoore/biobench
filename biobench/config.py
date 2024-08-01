from dataclasses import dataclass
from enum import Enum


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
    pressure: int

