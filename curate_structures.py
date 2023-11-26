# script to take the XYZ files and curate only valid structures from them
# criteria for valid structure:
#   - has exactly 4 pyrole rings - ensure no dimers or other weird structures
#   - has exactly 3 meso carbons for corroles and 4 meso carbons for porphyrins - ensure we deal with a corrole / porphyrin macrocycle
#   - has exactly one metal center
import os
import shutil
import utils, config

def curate_corroles():
    valid_mols = 0
    total_mols = 0
    for fname in os.listdir(config.CORROLE_XYZ_DIR):
        total_mols += 1
        mol = utils.get_molecule(config.CORROLE_XYZ_DIR + fname)
        nitrogens = utils.find_pyrrolic_nitrogens(mol)
        if len(nitrogens) == 4:
            metal, metas, betas = utils.disect_complex(mol)
            if len(metas) == 3 and not metal == None:
                valid_mols += 1
                shutil.copyfile(config.CORROLE_XYZ_DIR + fname, config.CORROLE_CURATED_XYZ_DIR + fname)                
    print("TOTAL NUMBER =", total_mols)
    print("VALID MOLECULES =", valid_mols)

def curate_porphyrins():
    valid_mols = 0
    total_mols = 0
    for fname in os.listdir(config.PORPHYRIN_XYZ_DIR):
        total_mols += 1
        mol = utils.get_molecule(config.PORPHYRIN_XYZ_DIR + fname)
        nitrogens = utils.find_pyrrolic_nitrogens(mol)
        if len(nitrogens) == 4:
            print("1" * 30)
            try:
                metal, metas, betas = utils.disect_complex(mol)
            except IndexError:
                # in case the disection fails due to index issues, the molecule is probably not a porphyrin
                valid_mols += 1
                print("*" * 30)
                continue
            if len(metas) == 4:
                shutil.copyfile(config.PORPHYRIN_XYZ_DIR + fname, config.PORPHYRIN_CURATED_XYZ_DIR + fname)                
    print("TOTAL NUMBER =", total_mols)
    print("VALID MOLECULES =", valid_mols)

if __name__ == "__main__":
    curate_corroles()