import utils
import os
import torinax as tx
from torinax.utils.openbabel import ob_read_file_to_molecule
from rdkit.Chem import rdchem


BASE_INPUT_STRING = """! UKS B3LYP ma-def2-SVP OPT NumFreq
%basis
$BASIS_SET
end 

%scf
  maxiter 200
end

%pal 
	nprocs 16
end
"""

BASIS_SET_STR = "NewGTO $SYMBOL \"ma-def2-TZVP\" end"

def make_input_str(molecule: tx.base.Molecule):
    basis_strs = set()
    for atom in molecule.atoms:
        Z = rdchem.GetPeriodicTable().GetAtomicNumber(atom.symbol)
        if Z >= 35: 
            basis_strs.add(BASIS_SET_STR.replace("$SYMBOL", atom.symbol))
    ajr = "\n".join(basis_strs)
    print(ajr)
    return BASE_INPUT_STRING.replace("$BASIS_SET", ajr)


def single_calc(xyz_path, input_dir):
    """Method to make an sql entry and command line string for single SLURM run"""
    molecule = ob_read_file_to_molecule(xyz_path)
    name = os.path.split(xyz_path)[-1].split("_")[0]
    in_file_path = os.path.join(input_dir, "{}.inp".format(name))
    infile = tx.io.OrcaIn(in_file_path)
    kwds = {"input_text": make_input_str(molecule)}
    kwds["charge"] = 0
    # correcting for unique case of oxygen molecule - triplet ground state
    kwds["mult"] = 1
    infile.write_file(molecule, kwds)

def make_inputs(structure: str):
    cur_dir = utils.get_directory("curated", structure)
    dft_dir = os.path.join(utils.get_directory("dft", structure, create_dir=True), "inputs")
    for xyz in os.listdir(cur_dir):
        single_calc(os.path.join(cur_dir, xyz), dft_dir)

if __name__ == "__main__":
    args = utils.read_command_line_arguments("script to make ORCA DFT calculation inputs based on curated structures")
    make_inputs(args.structure)
