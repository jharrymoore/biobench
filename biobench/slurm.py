import os
import subprocess


class SlurmJob:
    def __init__(
        self,
        name,
        job_id,
        job_dir,
        time,
        account,
        partition,
        gres,
        ntasks,
        nodes,
        mem,
        command,
    ):
        self.name = name
        self.job_id = job_id
        self.job_dir = job_dir
        self.time = time
        self.account = account
        self.partition = partition
        self.gres = gres
        self.ntasks = ntasks
        self.nodes = nodes
        self.mem = mem
        self.command = command

    def create_slurm_header(self):

        return f"""#!/bin/bash -l
#SBATCH --time={self.time}
#SBATCH --partition={self.partition}
#SBATCH --gres={self.gres}
#SBATCH --account={self.account}
#SBATCH --ntasks={self.ntasks}
#SBATCH --nodes={self.nodes}
#SBATCH --mem={self.mem}

source ~/.bashrc
conda activate mace-mlmm"""

    def write_job_script(self):
        with open(f"{self.job_dir}/{self.name}.sh") as f:
            f.write(self.create_slurm_header())
            f.write("\n")
            f.write(self.command)

    def submit_job(self):
        subprocess.run()
