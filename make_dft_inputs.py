import os
import openbabel as ob
import torinax as tx
from torinax.utils.openbabel import molecule_to_obmol, ob_read_file_to_molecule
from rdkit.Chem import rdchem
import config


BASE_INPUT_STRING = """! UKS B3LYP ma-def2-SVP OPT NumFreq
%basis
$BASIS_SET
end 

%scf
  maxiter 200
end

%CPCM SMD TRUE
    SMDSOLVENT "ACETONITRILE"
END

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


def single_calc(xyz_path, input_dir, mult):
    """Method to make an sql entry and command line string for single SLURM run"""
    molecule = ob_read_file_to_molecule(xyz_path)
    name = os.path.split(xyz_path)[-1].split(".")[0]
    in_file_path = os.path.join(input_dir, "{}_{}.inp".format(name, mult))
    infile = tx.io.OrcaIn(in_file_path)
    kwds = {"input_text": make_input_str(molecule)}
    kwds["charge"] = 0
    # correcting for unique case of oxygen molecule - triplet ground state
    kwds["mult"] = mult
    infile.write_file(molecule, kwds)

def make_corrole_inputs():
    for xyz in os.listdir(config.CORROLE_CURATED_XYZ_DIR):
        single_calc(os.path.join(config.CORROLE_CURATED_XYZ_DIR, xyz), config.CORROLE_DFT_INPUT_DIR, 1)
        single_calc(os.path.join(config.CORROLE_CURATED_XYZ_DIR, xyz), config.CORROLE_DFT_INPUT_DIR, 3)

if __name__ == "__main__":
    make_corrole_inputs()
