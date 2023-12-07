import os
import openbabel as ob
from typing import List
import config

def get_molecule(path: str) -> ob.OBMol:
    """Method to get a molecule by its name"""
    obmol = ob.OBMol()
    conv = ob.OBConversion()
    conv.ReadFile(obmol, path)
    return obmol

def find_pyrrole_indices(mol: ob.OBMol) -> List[List[int]]:
    # matches the corrole ring by SMARTS matching
    # first, we match all the pyrrolic nitrogens (only aromatic nitrogens, rest are kekulized)
    smarts_pattern = ob.OBSmartsPattern()
    # smarts_pattern.Init("[n]")
    smarts_pattern.Init("c1cccn1")
    smarts_pattern.Match(mol)
    # building unique list of indices
    ajr = []
    for x in smarts_pattern.GetMapList():
        s = set(x)
        if s not in ajr:
            ajr.append(s)
    return [list(x) for x in ajr]
    
def find_pyrrolic_nitrogens(mol: ob.OBMol) -> List[ob.OBAtom]:
    ajr = []
    for atom_indices in find_pyrrole_indices(mol):
        for i in atom_indices:
            atom = mol.GetAtom(i)
            # if atom is nitrogen, add it to the results
            if atom.GetAtomicNum() == 7:
                ajr.append(atom)
    return ajr

def disect_complex(mol: ob.OBMol) -> List[List[ob.OBAtom]]:
    """Method to extract the indices of the beta carbons, meta carbons and central metal"""
    pyrrole_indices = find_pyrrole_indices(mol)
    nitrogens = find_pyrrolic_nitrogens(mol)
    # we analyze the first nearest neighbors - this is the metal and the atoms neighboring the meta and beta carbons
    carbon_neighbors = []
    metal_idxs = []
    for n in nitrogens:
        neighbors = [atom for atom in ob.OBAtomAtomIter(n)]
        metal_idxs += [atom.GetIdx() for atom in neighbors if atom.IsMetal()]
        carbon_neighbors += [atom for atom in neighbors if not atom.IsMetal()]
    # now we go to the second nearest neighbors of the carbon neighbors - these are the meta and beta carbons
    if len(metal_idxs) == 0:
        return None, [], []
    betas = set()
    metas = set()
    for c in carbon_neighbors:
        neighbors = [atom for atom in ob.OBAtomAtomIter(c)]
        for neighbor in neighbors:
            # beta carbon is the carbon neighbor that is in a pyrrolic ring
            if any([neighbor.GetIdx() in x for x in pyrrole_indices]) and neighbor.GetAtomicNum() == 6 and not neighbor.GetIdx() in [a.GetIdx() for a in carbon_neighbors]:
                betas.add(neighbor.GetIdx())
            # meta carbons are the neighbor outside of the ring
            elif all([neighbor.GetIdx() not in x for x in pyrrole_indices]):
                metas.add(neighbor.GetIdx())
    return metal_idxs[0], list(metas), list(betas)

def read_command_line_arguments(script_description: str="", return_args: bool=True):
    import argparse
    parser = argparse.ArgumentParser(description=script_description)
    parser.add_argument("structure", type=str, help="structure you want to use (corrole, porphyrin etc.)")
    if return_args:
        return parser.parse_args()
    else:
        return parser

def get_directory(ftype: str, structure: str, create_dir: bool=False):
    """get the proper directory of a filetype (cif, xyz, curated...) and structure"""
    ftype_path = os.path.join(config.DATA_DIR, ftype)
    if os.path.isdir(ftype_path):
        struct_dir = os.path.join(ftype_path, structure)
        if os.path.isdir(struct_dir):
            return struct_dir
        elif create_dir:
            os.mkdir(struct_dir)
        else:
            raise ValueError("Structure not available for this file type. existing structures " + ", ".join(os.listdir(ftype_path)))
    else:
        raise ValueError("File type not available. existing file types " + ", ".join(os.listdir(config.DATA_DIR)))
    
def clean_directory(direactory: str, extension: str):
    """Make sure that a direactory contains only files of shape {cod_id}.{extension} and nothing else"""
    for fname in os.listdir(direactory):
        cid = fname.split(".")[0]
        if "{}.{}".format(cid, extension) != fname:
            os.remove(os.path.join(direactory, fname))
