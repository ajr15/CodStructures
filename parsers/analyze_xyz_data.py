import os
from typing import List, Callable
import openbabel as ob
from rdkit.Chem import rdchem
import numpy as np
from scipy.linalg import lstsq
import pandas as pd
import utils
import config

def metal_bonding_analyzer(mol_path: str):
    """Characterize bonding of the metal. returns coordination number, mean M-N bond length and mean M-axial bond length"""
    mol = utils.get_molecule(mol_path)
    nitrogens = [a.GetIdx() for a in utils.find_pyrrolic_nitrogens(mol)]
    metal, metas, betas = utils.disect_complex(mol)
    metal_Z = mol.GetAtom(metal).GetAtomicNum()
    bonds = [bond for bond in ob.OBAtomBondIter(mol.GetAtom(metal))]
    metal_nitrogen = []
    metal_axial = []
    expected_axial_distance = []
    for bond in bonds:
        if any([bond.GetBeginAtomIdx() in nitrogens, bond.GetEndAtomIdx() in nitrogens]):
            metal_nitrogen.append(bond.GetLength())
        else:
            metal_axial.append(bond.GetLength())
            axial_atom_radius = rdchem.GetPeriodicTable().GetRvdw(bond.GetEndAtom().GetAtomicNum()) if bond.GetEndAtom().GetAtomicNum() != metal_Z else rdchem.GetPeriodicTable().GetRvdw(bond.GetBeginAtom().GetAtomicNum())
            expected_axial_distance.append(rdchem.GetPeriodicTable().GetRvdw(metal_Z) + axial_atom_radius)
    res = {"coordination": len(bonds), "metal_nitrogen_bond_length": np.mean(metal_nitrogen), "metal_axial_bond_length": np.mean(metal_axial), "expected_metal_axial_bl": np.mean(expected_axial_distance)}
    # adding some metadata on metal
    res["expected_metal_nitrogen_bl"] = rdchem.GetPeriodicTable().GetRvdw(metal_Z) + rdchem.GetPeriodicTable().GetRvdw(7)
    return res

def calculate_4n_plane(nitrogens: List[ob.OBAtom]) -> List[float]:
    """Calculate the best fitting plane going through the nitrogen atoms"""
    # collect coordinates
    coords = []
    for atom in nitrogens:
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    coords = np.array(coords)
    # now fitting a plane by solving the equation Ax = b (x are the plane parameters)
    A = np.column_stack((coords[:, 0], coords[:, 1], np.ones(len(nitrogens))))
    x, _, _, _ = lstsq(A, coords[:, 2])
    # the plane equation will be ax + by + c = z
    a, b, c = x
    # returning parameters according to the equation ax + by + cz + d = 0
    return np.array([a, b, -1, c])

def calculate_distance_from_plane(point: np.ndarray, plane_parameters: np.ndarray):
    """Calculate the distance of point from a plane. given a point coordinates and plane a, b, c, d parameters"""
    a, b, c, d = plane_parameters
    x, y, z = point
    return abs(a * x + b * y + c * z + d) / np.linalg.norm([a, b, c])

def calculate_non_planarity(atoms: List[ob.OBAtom], plane: np.ndarray):
    """Calculate the average distance from a plane of a collection of atoms"""
    distances = []
    for atom in atoms:
        point = np.array([atom.GetX(), atom.GetY(), atom.GetZ()])
        dist = calculate_distance_from_plane(point, plane)
        distances.append(dist)
    return np.mean(distances)


def planarity_analyzer(mol_path: str):
    """Analyze the planarity of the macrocycle. returns coordination number, mean M-N bond length and mean M-axial bond length"""
    mol = utils.get_molecule(mol_path)
    res = {}
    nitrogens = utils.find_pyrrolic_nitrogens(mol)
    plane = calculate_4n_plane(nitrogens)
    res["nitrogen_non_planarity"] = calculate_non_planarity(nitrogens, plane)
    metal, metas, betas = utils.disect_complex(mol)
    res["metal_non_planarity"] = calculate_non_planarity([mol.GetAtom(metal)], plane)
    res["metas_non_planarity"] = calculate_non_planarity([mol.GetAtom(a) for a in metas], plane)
    res["betas_non_planarity"] = calculate_non_planarity([mol.GetAtom(a) for a in betas], plane)
    return res

def find_counter_mols(mol_path: str):
    """Method to extract other molecules in the crystal structure. this is done mostly for best esitmation of complex charge"""
    return list(map(utils.mol_to_smiles, utils.get_counter_mols(mol_path)))

def basic_details_analyzer(mol_path: str):
    mol = utils.get_molecule(mol_path)
    res = {}
    # central metal info
    metal, metas, betas = utils.disect_complex(mol)
    if metal is not None:
        metal_Z = mol.GetAtom(metal).GetAtomicNum()
        res["metal"] = rdchem.GetPeriodicTable().GetElementSymbol(metal_Z)
    else:
        res["metal"] = None
    # n electrons info
    nelec = 0
    for atom in ob.OBMolAtomIter(mol):
        nelec += atom.GetAtomicNum()
    res["n_electrons"] = nelec
    # solvents and more
    res["solvent"] = find_counter_mols(mol_path)
    return res

def make_dataframe(analyzers: List[Callable], base_struct: str) -> pd.DataFrame:
    path = utils.get_directory("curated", base_struct)
    ajr = []
    for fname in os.listdir(path):
        sid = fname.split("_")[0]
        print(sid)
        mol = os.path.join(path, fname)
        d = {"sid": sid}
        for analyzer in analyzers:
            d.update(analyzer(mol))
        ajr.append(d)
    ajr = pd.DataFrame(data=ajr)
    ajr = ajr.set_index("sid")
    return ajr

if __name__ == "__main__":
    args = utils.read_command_line_arguments("parse basic data from curated XYZ files. mostly around geometries.")
    df = make_dataframe([basic_details_analyzer, metal_bonding_analyzer, planarity_analyzer], args.structure)
    df.to_csv(os.path.join(config.RESULTS_DIR, '{}_xyz_analysis.csv'.format(args.structure)))
