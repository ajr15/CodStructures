# script to take the XYZ files and curate only valid structures from them
# criteria for valid structure:
#   - has exactly 4 pyrole rings - ensure no dimers or other weird structures
#   - has exactly 3 meso carbons for corroles and 4 meso carbons for porphyrins - ensure we deal with a corrole / porphyrin macrocycle
#   - has exactly one metal center
import networkx as nx
import os
from functools import reduce
import shutil
import openbabel as ob
import multiprocessing
import utils

def topologically_valid(mol: ob.OBMol, structure: str, nisomorphs: int) -> bool:
    """Validate the topology of the complex"""
    isos = utils.find_structure_indices(mol, structure)
    # first check if the number of isomorphs is correct
    if len(isos) != nisomorphs:
        return False
    # make sure there are exactly 4 pyrrole rings in the structure
    ajr = utils.find_structure_indices(mol, "pyrrole")
    # first check if the number of isomorphs is correct
    if len(ajr) != 4:
        return False
    # then make sure the structure is properly saturated (exactly 3 neighbors for each atom)    
    macrocycle = list(set(reduce(lambda x, y: x + y, isos)))
    for idx in macrocycle:
        atom = mol.GetAtom(idx)
        if len(list(ob.OBAtomAtomIter(atom))) != 3:
            return False
    # finally, make sure that each meta and beta sub is separate
    metal, metas, betas = utils.disect_complex(mol)
    if metal is None:
        return False
    g = utils.mol_to_graph(mol)
    for idx in metas + betas:
        out_neighbor = None
        for n in g.neighbors(idx):
            if not n in macrocycle:
                out_neighbor = n
        g.remove_edge(idx, out_neighbor)
        if nx.is_connected(g):
            return False
        g.add_edge(idx, out_neighbor)
    return True


def valid_charge(mol_path: str, mol: ob.OBMol) -> bool:
    """Validate that the electronic structure of the complex is valid. we mainly want to make sure that the total charge of the complex is 0"""
    # make sure the number of electron is not odd
    nelecs = sum([a.GetAtomicNum() for a in ob.OBMolAtomIter(mol)])
    if nelecs % 2 == 1:
        return False
    # make sure all the counter molecules in the crystal structure do not have odd number of electrons
    for counter in utils.get_counter_mols(mol_path):
        nelecs = sum([a.GetAtomicNum() for a in ob.OBMolAtomIter(counter)])
        if nelecs % 2 == 1:
            return False
    return True
        
def curate_structure(args):
    mol_path, target_path, structure, nisomorphs = args
    mol = utils.get_molecule(mol_path)
    if not topologically_valid(mol, structure, nisomorphs):
        return
    if not valid_charge(mol_path, mol):
        return    
    # if it passed all the tests, copy the file to the curated directory
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
    mol_path = "data/xyz/corroles/4513497_0.xyz"
    structure = "corroles"
    nisomorphs = 1
    mol = utils.get_molecule(mol_path)
    isos = utils.find_structure_indices(mol, structure)
    # first check if the number of isomorphs is correct
    if len(isos) != nisomorphs:
        print(1)
    # make sure there are exactly 4 pyrrole rings in the structure
    ajr = utils.find_structure_indices(mol, "pyrrole")
    # first check if the number of isomorphs is correct
    if len(ajr) != 4:
        print(2)
    # then make sure the structure is properly saturated (exactly 3 neighbors for each atom)    
    macrocycle = list(set(reduce(lambda x, y: x + y, isos)))
    for idx in macrocycle:
        atom = mol.GetAtom(idx)
        if len(list(ob.OBAtomAtomIter(atom))) != 3:
            print(3)
    # finally, make sure that each meta and beta sub is separate
    metal, metas, betas = utils.disect_complex(mol)
    print(len(metas), len(betas))
    g = utils.mol_to_graph(mol)
    for idx in metas + betas:
        out_neighbor = None
        for n in g.neighbors(idx):
            if not n in macrocycle:
                out_neighbor = n
        g.remove_edge(idx, out_neighbor)
        if nx.is_connected(g):
            print(4)
        g.add_edge(idx, out_neighbor)


if __name__ == "__main__":
    # test_corrole()
    # exit()
    parser = utils.read_command_line_arguments("curate XYZ files using substructure matching", return_args=False)
    parser.add_argument("--nworkers", type=int, default=1, help="number of worker for parallel processing of files")
    parser.add_argument("--nisomorphs", type=int, default=1, help="number of distinct isomorphisms between definition and molecule (1=monomer, 2=dimer...)")
    args = parser.parse_args()
    main(args.structure, args.nisomorphs, args.nworkers)