# script to take the XYZ files and curate only valid structures from them
# criteria for valid structure:
#   - has exactly 4 pyrole rings - ensure no dimers or other weird structures
#   - has exactly 3 meso carbons for corroles and 4 meso carbons for porphyrins - ensure we deal with a corrole / porphyrin macrocycle
#   - has exactly one metal center
import os
import shutil
import multiprocessing
import utils

def curate_structure(args):
    mol_path, target_path, structure, nisomorphs = args
    mol = utils.get_molecule(mol_path)
    if utils.validate_structure(mol, structure, nisomorphs):
        shutil.copy(mol_path, target_path)

def main(structure: str, nisomorphs: int, nworkers: int):
    print("initializing...")
    xyz_dir = utils.get_directory("xyz", structure)
    cur_dir = utils.get_directory("curated", structure, create_dir=True)
    args = []
    for fname in os.listdir(xyz_dir):
        args.append((os.path.join(xyz_dir, fname), os.path.join(cur_dir, fname), structure, nisomorphs))
    # running parallel the conversion jobs
    print("starting conversion...")
    if nworkers > 1:
        with multiprocessing.Pool(nworkers) as pool:
            pool.map(curate_structure, args)
    else:
        list(map(curate_structure, args))
    # cleaning garbage files 
    print("cleaning garbage...")
    utils.clean_directory(cur_dir, "xyz")
    print("ALL DONE!")
    print("total curated XYZ files:", len(os.listdir(cur_dir)))


def test_corrole():
    fname = "7042334_0.xyz"
    mol = utils.get_molecule(utils.get_directory("xyz", "porphyrins") + fname)
    # mol = utils.get_molecule(os.path.join(config.DATA_DIR, "definitions", "corrole.mol"))
    print(utils.validate_structure(mol, "porphyrins", 1))


if __name__ == "__main__":
    # test_corrole()
    # exit()
    parser = utils.read_command_line_arguments("curate XYZ files using substructure matching", return_args=False)
    parser.add_argument("--nworkers", type=int, default=1, help="number of worker for parallel processing of files")
    parser.add_argument("--nisomorphs", type=int, default=1, help="number of distinct isomorphisms between definition and molecule (1=monomer, 2=dimer...)")
    args = parser.parse_args()
    main(args.structure, args.nisomorphs, args.nworkers)