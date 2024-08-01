from dataclasses import dataclass


@dataclass
class MaceMDCLI:
    pressure: float
    temp: float
    file: str
    ml_mol: str
    output_dir: str
    model_path: str
    steps: int
    run_type: str = "md"
    replicas: int = 1
    resname: str = "LIG"
    decouple: bool = False
    padding: float = 0.0
    box_shape: str = "cube"

    def create_cli_string(self) -> str:
        return f"""mace-md --pressure {self.pressure} \\
    --temperature {self.temp} \\
    --file {self.file} \\
    --ml_mol {self.ml_mol} \\
    --output_dir {self.output_dir} \\
    --model_path {self.model_path} \\
    --steps {self.steps} \\
    --run_type {self.run_type} \\
    --replicas {self.replicas} \\
    --padding {self.padding} \\
    --box_shape {self.box_shape} \\
"""
