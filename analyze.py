import os
from typing import List
import openbabel as ob
from rdkit.Chem import rdchem
import numpy as np
from scipy.linalg import lstsq
import pandas as pd
import utils
import config

def analyze_metal_bonding(mol: ob.OBMol):
    """Characterize bonding of the metal. returns coordination number, mean M-N bond length and mean M-axial bond length"""
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
    res["metal"] = rdchem.GetPeriodicTable().GetElementSymbol(metal_Z)
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


def analyze_planarity(mol: ob.OBMol):
    """Analyze the planarity of the macrocycle. returns coordination number, mean M-N bond length and mean M-axial bond length"""
    res = {}
    nitrogens = utils.find_pyrrolic_nitrogens(mol)
    print([n.GetIdx() for n in nitrogens])
    plane = calculate_4n_plane(nitrogens)
    res["nitrogen_non_planarity"] = calculate_non_planarity(nitrogens, plane)
    metal, metas, betas = utils.disect_complex(mol)
    res["metal_non_planarity"] = calculate_non_planarity([mol.GetAtom(metal)], plane)
    res["metas_non_planarity"] = calculate_non_planarity([mol.GetAtom(a) for a in metas], plane)
    res["betas_non_planarity"] = calculate_non_planarity([mol.GetAtom(a) for a in betas], plane)
    return res

if __name__ == "__main__":
    df = pd.DataFrame()
    for fname in os.listdir(config.CORROLE_CURATED_XYZ_DIR):
        sid = fname[:-4]
        print(sid)
        mol = utils.get_molecule(os.path.join(config.CORROLE_CURATED_XYZ_DIR, fname))
        d = analyze_metal_bonding(mol)
        d.update(analyze_planarity(mol))
        d["id"] = sid
        df = df.append(d, ignore_index=True)
    df.to_csv("metal_data.csv")
