from dataclasses import dataclass


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
