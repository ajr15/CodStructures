from typing import List, Callable
import pandas as pd
import os
import numpy as np
from scipy.linalg import lstsq
import openbabel as ob
from networkx import isomorphism
import utils

def calculate_4n_plane(atoms: List[ob.OBAtom]) -> List[float]:
    """Calculate the best fitting plane going through the nitrogen atoms"""
    # collect coordinates
    coords = []
    for atom in atoms:
        if atom.GetAtomicNum() == 7:
            coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    coords = np.array(coords)
    # now fitting a plane by solving the equation Ax = b (x are the plane parameters)
    A = np.column_stack((coords[:, 0], coords[:, 1], np.ones(len(coords))))
    x, _, _, _ = lstsq(A, coords[:, 2])
    # the plane equation will be ax + by + c = z
    a, b, c = x
    # returning parameters according to the equation ax + by + cz + d = 0
    return a, b, c

def atoms_to_coordinates(atoms):
    coords = []
    for atom in atoms:
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    return np.array(coords)

def find_z_coordinates(atoms: List[ob.OBAtom]) -> np.ndarray:
    """method to find z-coordinates of points from best-fitting plane"""
    points = atoms_to_coordinates(atoms)
    x = points[:, 0]
    y = points[:, 1]
    a, b, c = calculate_4n_plane(atoms)
    # calculating z values from equation
    return - (a * x + b * y + c)


def get_substracture_mapping(obmol: ob.OBMol, structure: str) -> dict:
    """Get the substructure index mapping between a molecule and substructure"""
    subgraph = utils.get_definition(structure)
    g = utils.mol_to_graph(obmol)
    iso = isomorphism.GraphMatcher(g, subgraph, node_match=utils.node_matcher)
    # returning the first morph. return the reversed mapping (subgraph -> mol)
    for morph in iso.subgraph_isomorphisms_iter():
        return {v: k for k, v in morph.items()}


def match_substructures(mol1: ob.OBMol, mol2: ob.OBMol, structure: str):
    """Method to match two different molecules' substructures. i.e. match the macrocycle indices between the molecules. returns two lists of matching indices"""
    morph1 = get_substracture_mapping(mol1, structure)
    morph2 = get_substracture_mapping(mol2, structure)
    return list(morph1.values()), [morph2[k] for k in morph1.keys()]


def idxs_to_atoms(mol: ob.OBMol, idxs) -> List[ob.OBAtom]:
    """Method to convert list of atom indices in molecule to list of points in 3d space"""
    coords = []
    for idx in idxs:
        atom = mol.GetAtom(idx)
        coords.append(atom)
    return coords


def non_planarity_mod_analyzer(mol: str, displaced_structures: dict, base_struct: str):
    """Use cosine similarity to check the non-planarity type of a molecule"""
    ajr = {}
    mol = utils.get_molecule(mol)
    for name, struct in displaced_structures.items():
        # get matching displacement vectors
        delta1, delta2 = map(find_z_coordinates, [idxs_to_atoms(m, i) for m, i in zip([mol, struct], match_substructures(mol, struct, base_struct))])
        delta2 = delta2 / np.linalg.norm(delta2)
        # calculate cosine similarity
        ajr[name] = abs(np.sum(delta1 * delta2) / (np.linalg.norm(delta1) * np.linalg.norm(delta2)))
    ajr["total_displacement"] = np.linalg.norm(delta1)
    return ajr    


def compare_displaced_structures(base_struct: str, displaced_structures=None):
    if displaced_structures is None:
        displaced_structures = utils.get_displaced_structures(base_struct)
    data = np.zeros((len(displaced_structures), len(displaced_structures)))
    for i, struct1 in enumerate(displaced_structures.values()):
        for j, struct2 in enumerate(displaced_structures.values()):
            # get matching displacement vectors
            delta1, delta2 = map(find_z_coordinates, [idxs_to_atoms(m, i) for m, i in zip([struct1, struct2], match_substructures(struct1, struct2, base_struct))])
            data[i, j] = np.sum(delta1 * delta2) / (np.linalg.norm(delta1) * np.linalg.norm(delta2))
    return pd.DataFrame(data=data, columns=displaced_structures.keys(), index=displaced_structures.keys())


def uncorrelated_structures(base_struct: str, th: float):
    structs = utils.get_displaced_structures(base_struct)
    df = compare_displaced_structures(base_struct, structs).abs()
    df = df[sorted(df.columns)]
    df = df.sort_index()
    df = df >= th
    unique = set([df.index.values[df[col].to_list().index(True)] for col in df.columns])
    return {key: structs[key] for key in unique}



def make_dataframe(analyzers: List[Callable], base_struct: str, displaced_structures=None) -> pd.DataFrame:
    path = utils.get_directory("curated", base_struct)
    if displaced_structures is None:
        displaced_structures = utils.get_displaced_structures(base_struct)
    ajr = []
    for fname in os.listdir(path):
        sid = fname.split("_")[0]
        print(sid)
        mol = os.path.join(path, fname)
        d = {"sid": sid}
        for analyzer in analyzers:
            d.update(analyzer(mol, displaced_structures, base_struct))
        ajr.append(d)
    ajr = pd.DataFrame(data=ajr)
    ajr = ajr.set_index("sid")
    return ajr


if __name__ == "__main__":
    import config
    import matplotlib.pyplot as plt
    args = utils.read_command_line_arguments("parse basic data from curated XYZ files. mostly around geometries.")
    df = compare_displaced_structures(args.structure)
    df = df[sorted(df.columns)]
    df = df.sort_index()
    print(df.abs())
    plt.matshow(df)
    plt.xticks(range(len(df.columns)), df.columns, rotation=45)
    plt.yticks(range(len(df.columns)), df.index)
    plt.title("Cosine Similarity of Reference Structures")
    plt.colorbar()
    exit()
    displaced_structs = uncorrelated_structures(args.structure, 0.5)
    df = make_dataframe([non_planarity_mod_analyzer], args.structure, displaced_structs)
    df.to_csv(os.path.join(config.RESULTS_DIR, '{}_non_planarity_analysis.csv'.format(args.structure)))
    # normalize dataframe
    plt.figure()
    for col in df.columns:
        if col != "total_displacement":
            plt.hist(df[col], label=col, alpha=0.6, bins=50)
    sorted_cols = sorted([c for c in df.columns if not c == "total_displacement"])
    sorted_cols.append("total_displacement")
    df = df[sorted_cols]
    plt.legend()
    plt.title("Distribution of Metrics")
    plt.figure()
    plt.hist(df["total_displacement"])
    plt.title("Total Displacement Distribution")
    print(df)
    plt.matshow(df.corr() ** 2, cmap="Greens")
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="left")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title("Correlation in Sample (R^2)")
    plt.colorbar()
    plt.show()
