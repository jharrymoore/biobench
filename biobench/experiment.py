import os
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import shutil
import mdtraj as md
import tempfile
from typing import List, Optional, Tuple
from urllib.request import urlretrieve
import tarfile
from pdbfixer.pdbfixer import PDBFixer
from biobench.mace_md import MaceMDCLI
from openmm import unit
from openmm import app
from result import Ok, Err, Result
from pymbar import timeseries
import logging
from mace.modules.models import MACE
from copy import deepcopy
from cuda_mace.models import EquivariantMACE
from biobench.config import SlurmParams, ExecutionEnvironment, ExperimentResult, SimulationParams
import mpiplus
from mace_md.hybrid_md import PureSystem
import torch

logger = logging.getLogger("BIOBENCH")

class Experiment:
    data_path: str
    output_dir: str
    model: MACE
    partition: str
    slurm_params: Optional[SlurmParams]

    def __init__(
        self,
        data_path: str,
        model_path: str,
        overwrite: bool,
        steps: int,
        optimize_model: bool,
        execution_environment: ExecutionEnvironment,
        run_max_n: int,
        state_dict: Optional[str] = None,
        slurm_params: Optional[SlurmParams] = None,

    ):
        self.workdir = tempfile.mkdtemp()
        self.data_path = data_path
        self.model = torch.load(model_path)
        self.model_path=model_path
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.slurm_params = slurm_params
        self.overwrite = overwrite
        self.steps = steps
        self.execution_environment = execution_environment
        self.run_max_n = run_max_n
        if optimize_model:
            match self.optimize_model():
                case Ok((model, path)):
                    logger.info("Successfully equipped model with CUDA kernels")
                    self.model, self.model_path = model, path
                case Err(val):
                    logger.critical(f"Error in converting model to cuda kernels: {val}. Model will not be run with optimizations")
                    

    def download_data(self,):
        # download data from url to a tmp directory
        logging.info(f"Downloading data from {self.data_path}")
        urlretrieve(self.data_path, os.path.join(self.workdir, "data.tar.gz"))
        file_size = os.path.getsize(os.path.join(self.workdir, "data.tar.gz"))
        logging.info(f"Downloaded data size: {file_size} bytes")


    def unzip_data(self, workdir: str):
        data_file = os.listdir(workdir)[0]
        assert data_file.endswith(".tar.gz"), "Data file must be a tar.gz file."
        logging.info("Extracting tar.gz file...")
        with tarfile.open(os.path.join(workdir, data_file), "r:gz") as tar_ref:
            tar_ref.extractall(workdir)
        os.remove(os.path.join(workdir, "data.tar.gz"))
        extract_dir = os.path.join(workdir, os.listdir(workdir)[0])
        n_files = len(os.listdir(extract_dir))
        logging.info(f"Extracted {n_files} files.")
        if self.run_max_n != -1:
            for dir in os.listdir()
        return extract_dir

    def optimize_model(self) -> Result[Tuple[EquivariantMACE, str],str]:
        logging.info("Converting model to CUDA implementation...")
        # check whether we are L=0 or L=1
        model_lmax = self.model.interactions[1].node_feats_irreps.lmax
        if model_lmax == 0:
            # raise NotImplementedError("L=0 models are not currently torch scriptable")
            return Err("L=0 models are not currently torch scriptable")
        elif model_lmax == 1:
            model = EquivariantMACE(deepcopy(self.model))
            output_file = os.path.join(self.workdir, "mace_opt.model")
            torch.save(model, output_file)
            return Ok((model, output_file))
            
        
    def prepare_data(self):
        logging.info("Preparing data...")
        self.download_data()
        extract_dir = self.unzip_data()
        return extract_dir

    def execute(self):
        raise NotImplementedError("Execution should be implemented in the subclass")

    def analyse(self):
        raise NotImplementedError("Analysis should be implemented in the subclass")


class DensityExperiment(Experiment):
    def __init__(
        self,
        data_path: str,
        model_path: str,
        overwrite: bool,
        steps: int,
        execution_environment: ExecutionEnvironment,
        simulation_params: SimulationParams,
        csv_file:str, 
        slurm_params: Optional[SlurmParams] = None,
        optimize_model: bool = True,
        run_max_n: int = -1
    ):
        if execution_environment == ExecutionEnvironment.SLURM:
            assert slurm_params is not None, "Slurm job paramets not provided"
        super().__init__(
            data_path=data_path,
            model_path=model_path,
            overwrite=overwrite,
            steps=steps,
            execution_environment=execution_environment,
            optimize_model=optimize_model,
            run_max_n=run_max_n,
        )
        self.csv_file = csv_file
        self.simulation_params = simulation_params

    def execute(self) -> Result[None, str]:
        # if self.overwrite:
        #     logging.info("Overwriting existing data...")
        #     shutil.rmtree(self.workdir)
        # os.makedirs(self.workdir, exist_ok=True)
        extract_dir = self.prepare_data()

        def _run_mace_md(directory:str):
            local_mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK")
            print(f"Running simulation { directory} on rank {local_mpi_rank}")
            full_work_dir = os.path.join(self.workdir, extract_dir, directory)
            file = [os.path.join(self.workdir, extract_dir, directory, f) for f in os.listdir(os.path.join(self.workdir, extract_dir, directory)) if f.endswith(".xyz")]
            assert len(file) == 1, "More than one xyz file found"
            file = file[0]
            system = PureSystem(
                file=file,
                model_path=self.model_path,
                output_dir=os.path.join(full_work_dir,  "mace_md_liquid"),
                temperature=self.simulation_params.temp,
                minimiser=self.simulation_params.minimiser,
                optimized_model=self.simulation_params.optimized_model,
            )


            system.propagate(
                self.simulation_params.steps,
                output_file=os.path.join(os.path.dirname(file), "liquid_md"),
                interval=1000,
                restart=False
            )
        
        # distribute simulations across MPI ranks
        mpiplus.distribute(_run_mace_md, list(os.listdir(extract_dir)))

        return Ok(None)

    def analyse(self, exp_key: str, mol_name_key: str) -> Result[List[ExperimentResult], str]:
        logging.info("Analysing density data...")
        df = pd.read_csv(self.csv_file)
        output_dirs = os.listdir(self.output_dir)
        results = []
        # these must match molecule names in the dataframe
        for direc in output_dirs:
            traj = md.load(
                os.path.join(self.output_dir, direc, "output.dcd"),
                top=os.path.join(self.output_dir, direc, "output.pdb"),
            )
            density = md.density(traj) / 1000
            t0, g, Neff_max = timeseries.detect_equilibration(A_t) # compute indices of uncorrelated timeseries
            density_equil = density[t0:]
            logger.debug(f"Detected equilibration at {t0} of {len(density)}")
            logger.debug(f"Density for compound {direc}={density_equil.mean()}")
            results.append(
                ExperimentResult(
                    system_name=direc,
                    result=density_equil.mean(),
                    experimental=float(df[df[mol_name_key] == direc][exp_key]),
                )
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
