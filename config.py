COD_SEARCH_API_ENDPOINT = "https://www.crystallography.net/cod/result"
COD_FILE_API_ENDPOINT = "https://www.crystallography.net/cod/$CODID.cif"
import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = PROJECT_DIR + "/data/"
RESULTS_DIR = PROJECT_DIR + "/results/"
DISPLACED_STRUCTS_DIR = PROJECT_DIR + "/displaced_structures/"

CORROLE_RAW_JSON = DATA_DIR + "corrole_structures.json"
PORPHYRIN_RAW_JSON = DATA_DIR + "porphyrin_structures.json"
