import os
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import shutil
import mdtraj as md
import tempfile
from typing import Optional
from urllib.request import urlretrieve
import tarfile
from pdbfixer.pdbfixer import PDBFixer

from biobench.mace_md import MaceMDCLI
from biobench.slurm import SlurmJob
from openmm import unit
from openmm import app


class Experiment:
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        model_path: str,
        partition: str,
        timelimit: str,
        account: str,
        overwrite: bool,
        steps: int,
    ):
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_path = model_path
        self.partition = partition
        self.timelimit = timelimit
        self.account = account
        self.overwrite = overwrite
        self.steps = steps

    def download_data(self, path: str):
        # download data from url to a tmp directory
        logging.info(f"Downloading data from {self.data_path}")
        urlretrieve(self.data_path, os.path.join(path, "data.tar.gz"))
        file_size = os.path.getsize(os.path.join(path, "data.tar.gz"))
        logging.info(f"Downloaded data size: {file_size} bytes")

    def unzip_data(self, path: str):
        data_file = os.listdir(path)[0]
        assert data_file.endswith(".tar.gz"), "Data file must be a tar.gz file."
        logging.info("Extracting tar.gz file...")
        with tarfile.open(os.path.join(path, data_file), "r:gz") as tar_ref:
            tar_ref.extractall(path)
        os.remove(os.path.join(path, "data.tar.gz"))
        extract_dir = os.path.join(path, os.listdir(path)[0])
        n_files = len(os.listdir(extract_dir))
        logging.info(f"Extracted {n_files} files.")
        return extract_dir

    def prepare_data(self):
        logging.info("Preparing data...")
        self.download_data(self.output_dir)
        extract_dir = self.unzip_data(self.output_dir)
        return extract_dir

    def execute(self):
        raise NotImplementedError("Execution should be implemented in the subclass")

    def analyse(self):
        raise NotImplementedError("Analysis should be implemented in the subclass")


class DensityExperiment(Experiment):
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        model_path: str,
        partition: str,
        timelimit: str,
        account: str,
        overwrite: bool,
        steps: int,
        n_procs: Optional[int] = None,
        csv_file: Optional[str] = None,
    ):
        super().__init__(
            data_path=data_path,
            output_dir=output_dir,
            model_path=model_path,
            partition=partition,
            timelimit=timelimit,
            account=account,
            overwrite=overwrite,
            steps=steps,
        )
        self.n_procs = n_procs
        self.csv_file = csv_file

    def execute(self):
        if self.overwrite:
            logging.info("Overwriting existing data...")
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        extract_dir = self.prepare_data()

        for file in os.listdir(extract_dir):
            logging.info(f"Processing file: {file}")
            xyz_file = [
                f
                for f in os.listdir(os.path.join(extract_dir, file))
                if f.endswith(".xyz")
            ][0]
            command = MaceMDCLI(
                pressure=1.0,
                temp=298.0,
                file=xyz_file,
                ml_mol=xyz_file,
                output_dir="mace_md_density",
                model_path=self.model_path,
                steps=self.steps,
            )
            cli_string = command.create_cli_string()
            slurm_job = SlurmJob(
                name=file,
                job_dir=os.path.join(extract_dir, file),
                partition=self.partition,
                time=self.timelimit,
                account=self.account,
                gres="gpu:1",
                ntasks=1,
                nodes=1,
                mem="16G",
                command=cli_string,
            )
            slurm_job.write_job_script()
            slurm_job.submit_job()

    def analyse(self):
        logging.info("Analysing density data...")
        # implement analysis here
        # read names and experimental values from csv file
        molecules, exp_values = [], []
        assert (
            self.csv_file is not None
        ), "CSV file must be provided, since analysis was called"
        with open(self.csv_file, "r") as f:
            data = f.readlines()
        for line in data:
            name, value = line.split(",")
            molecules.append(name)
            exp_values.append(float(value))

        # loop over the output directories, compute densities
        output_dirs = os.listdir(self.output_dir)
        for direc in output_dirs:
            traj = md.load(
                os.path.join(self.output_dir, direc, "output.dcd"),
                top=os.path.join(self.output_dir, direc, "output.pdb"),
            )


class SolvationExperiment(Experiment):
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        model_path: str,
        partition: str,
        timelimit: str,
        account: str,
        overwrite: bool,
        steps: int,
        replicas: int,
        n_gpu: int,
        n_procs: Optional[int] = None,
    ):
        super().__init__(
            data_path=data_path,
            output_dir=output_dir,
            model_path=model_path,
            partition=partition,
            timelimit=timelimit,
            account=account,
            overwrite=overwrite,
            steps=steps,
        )
        self.replicas = replicas
        self.n_procs = n_procs
        self.n_gpu = n_gpu

    def prepare_solvated_molecule(self, name: str) -> None:
        logging.debug(f"Preparing solvated molecule for {name}")
        # prepare solvated molecule
        fixer = PDBFixer(os.path.join(self.output_dir, name, f"{name}.pdb"))
        fixer.addSolvent(padding=1.2 * unit.nanometer, boxShape="dodecahedron")

        with open(os.path.join(self.output_dir, name, "solvated.pdb"), "w") as f:
            logging.debug("Writing output...")
            if fixer.source is not None:
                f.write("REMARK   1 PDBFIXER FROM: %s\n" % fixer.source)
            app.PDBFile.writeFile(fixer.topology, fixer.positions, f, True)

    def prepare_data(self) -> str:
        logging.info("Preparing solvation data...")
        # assume we are provided with a smiles file containing the solute
        if self.data_path.endswith(".smi"):
            logging.info("Solute file provided in smiles format")
            with open(self.data_path, "r") as f:
                smiles = f.readlines()
            # file should be formatted in two columns: name smi
            for line in smiles:
                name, smi = line.split()
                logging.info(f"Processing solute: {name}")
                os.makedirs(os.path.join(self.output_dir, name), exist_ok=True)
                # write solute in pdb format to the output directory
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                Chem.MolToPDBFile(
                    mol, os.path.join(self.output_dir, name, f"{name}.pdb")
                )
                self.prepare_solvated_molecule(name)

        return self.output_dir

    def execute(self):
        logging.info("Preparing solvation data...")
        extract_dir = self.prepare_data()
        print(extract_dir)
        logging.info("Executing solvation calculations...")
        pdb_file = "solvated.pdb"
        command = MaceMDCLI(
            pressure=1.0,
            temp=298.0,
            file=pdb_file,
            ml_mol=pdb_file,
            output_dir="solvation_repex",
            model_path=self.model_path,
            steps=self.steps,
            run_type="repex",
            replicas=self.replicas,
            resname="UNL",
        )

        cli_string = command.create_cli_string()
        for f in os.listdir(extract_dir):
            slurm_job = SlurmJob(
                name=f,
                job_dir=os.path.join(extract_dir, f),
                partition=self.partition,
                time=self.timelimit,
                account=self.account,
                n_gpu=self.n_gpu,
                ntasks=self.replicas,
                nodes=1,
                mem="16G",
                command=cli_string,
            )
            slurm_job.write_job_script()
            # slurm_job.submit_job()

    def analyse(self):
        logging.info("Analysing solvation data...")
        # implement analysis here
        pass
