# script to parse the CIF files achieved from COD into molecule files (XYZ or MOL) of the porphyrinoids
import os
from pymatgen.core import Structure
from pymatgen.analysis.local_env import JmolNN as AnalyzerNN
import multiprocessing
import signal
from contextlib import contextmanager
import config

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def find_molecules(struct: Structure):
    # running function with limit of 5 minutes - sometime the run gets stuck for some reason
    with time_limit(5 * 60):
        # preprocessing structure to ensure no partial occupancies
        if not struct.is_ordered:
            to_remove = []
            for i, site in enumerate(struct.sites):
                if all([x < 0.5 for x in site.species.as_dict().values()]):
                    to_remove.append(i)
            struct.remove_sites(to_remove)
            # if still the structure is not ordered, return empty list
            if not struct.is_ordered:
                return []
        # analyzing nearest neighbohrs using pymatgen
        nn_analyzer = AnalyzerNN()
        structure_graph = nn_analyzer.get_bonded_structure(structure=struct)
        return structure_graph.get_subgraphs_as_molecules()

def cif_to_xyz(args):
    cif_file, xyz_dir = args
    print("converting", cif_file)
    filename = os.path.split(cif_file)[-1][:-4]
    xyz_file = os.path.join(xyz_dir, filename + ".xyz")
    if not os.path.isfile(xyz_file):
        try:
            struct = Structure.from_file(cif_file)
            molecules = find_molecules(struct)
        except TimeoutException:
            print("timeout occured at", cif_file)
            return
        except Exception:
            print("errors occured at", cif_file)
            return
        if len(molecules) == 0:
            print("no molecules found in", cif_file)
            return
        # taking the largest molecule as the porphyrinoid molecule
        sizes = [len(x.sites) for x in molecules]
        mol = molecules[sizes.index(max(sizes))]
        # saving mol to xyz file
        mol.to(xyz_file, fmt="xyz")

if __name__ == "__main__":
    # collecting all inputs to convert
    args = []
    for fname in os.listdir(config.CORROLE_CIF_DIR):
        args.append((os.path.join(config.CORROLE_CIF_DIR, fname), config.CORROLE_XYZ_DIR))
    # running parallel the conversion jobs
    with multiprocessing.Pool(6) as pool:
        pool.map(cif_to_xyz, args)