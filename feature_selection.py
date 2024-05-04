from functools import reduce
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from scipy import stats
import pandas as pd
from openbabel import openbabel as ob
from featurizers import StructurePropertyFeaturizer, SubstituentPropertyFeaturizer, FunctionFeaturizer, Featurizer
from read_to_sql import Substituent, StructureProperty
import utils

def metal_radius(session, sid: int):
    """Get the VDW radius of the metal center"""
    metal = session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "metal").all()[0][0]
    metal = utils.mol_from_smiles(metal).GetAtom(1)
    return ob.GetVdwRad(metal.GetAtomicNum())

def _non_planarity_helper(session, sid, mode, units):
    v = session.query(StructureProperty.value).filter(StructureProperty.structure == sid).filter(StructureProperty.property == mode).filter(StructureProperty.units == units).all()
    return abs(v[0][0])

def non_planarity(mode, units):
    s = mode if "total" in mode else mode + " non planarity"
    return lambda session, sid: _non_planarity_helper(session, sid, s, units)

def _dominant_mode_helper(session, sid, th, mode):
    structprops = session.query(StructureProperty).\
                filter(StructureProperty.structure == sid).\
                filter(StructureProperty.property.contains("non planarity")).\
                filter(StructureProperty.units == "%").all()
    total_np = _non_planarity_helper(session, sid, "total out of plane (exp)", "A")
    max_mode = structprops[0]
    for sp in structprops[1:]:
        if max_mode.value < sp.value:
            max_mode = sp
    if total_np >= th:
        return max_mode.property.split()[0] == mode
    else:
        return False

def dominant_mode(mode: str, th: float=1):
    """featurizer to check the dominant mode of a molecule"""
    return lambda session, sid: _dominant_mode_helper(session, sid, th, mode)

# BETA_POS = ["beta" + str(i + 1) for i in range(8)]
# MESO_POS = ["meso" + str(i + 1) for i in range(4)]
# MACROCYCLE_POSITIONS = [BETA_POS[(2 * i):] + BETA_POS[:(2 * i)] + MESO_POS[i:] + MESO_POS[:i] for i in range(4)]
MACROCYCLE_POSITIONS = ["meso1", "beta1", "beta2", "meso2", "beta3", "beta4", "meso3", "beta5", "beta6", "meso4", "beta7", "beta8"]
MACROCYCLE_POSITIONS = [MACROCYCLE_POSITIONS[(3 * i):] + MACROCYCLE_POSITIONS[:(3 * i)] for i in range(4)]
AXIAL_POSITIONS = [["axial1", "axial2"], ["axial2", "axial1"]]

FEATURIZERS = {
    "cone_angles": SubstituentPropertyFeaturizer("cone angle", None, MACROCYCLE_POSITIONS[0] + AXIAL_POSITIONS[0], navalue=-1) +\
          FunctionFeaturizer("metal_radius", metal_radius, navalue=None),
    "vdw_distances": SubstituentPropertyFeaturizer("vdw nn dist", None, MACROCYCLE_POSITIONS[0], navalue=None) +\
          SubstituentPropertyFeaturizer("cone angle", None, AXIAL_POSITIONS[0], navalue=-1) + FunctionFeaturizer("metal_radius", metal_radius, navalue=None),
    "covalent_distances": SubstituentPropertyFeaturizer("covalent nn dist", None, MACROCYCLE_POSITIONS[0], navalue=None) +\
          SubstituentPropertyFeaturizer("cone angle", None, AXIAL_POSITIONS[0], navalue=-1) + FunctionFeaturizer("metal_radius", metal_radius, navalue=None),
    "nn_distances": SubstituentPropertyFeaturizer("None nn dist", None, MACROCYCLE_POSITIONS[0], navalue=None) +\
          SubstituentPropertyFeaturizer("cone angle", None, AXIAL_POSITIONS[0], navalue=-1) + FunctionFeaturizer("metal_radius", metal_radius, navalue=None),
}

REGRESSION_TARGETS = {
    "outer_homa": StructurePropertyFeaturizer(["outer_circuit homa"], [None], navalue=None),
    "inner_homa": StructurePropertyFeaturizer(["inner_circuit homa"], [None], navalue=None),
    "total_out_of_plane": FunctionFeaturizer("abs. ruffling", non_planarity("total out of plane (exp)", "A"), navalue=None),
    "abs_ruffling": FunctionFeaturizer("abs. ruffling", non_planarity("ruffling", "A"), navalue=None),
    "abs_saddling": FunctionFeaturizer("abs. saddling", non_planarity("saddling", "A"), navalue=None),
    "abs_doming": FunctionFeaturizer("abs. doming", non_planarity("doming", "A"), navalue=None),
}

CLASSIFICATION_TARGETS = {
    "saddling": FunctionFeaturizer("saddling", dominant_mode("saddling"), navalue=None),
    "ruffling": FunctionFeaturizer("ruffling", dominant_mode("ruffling"), navalue=None),
    "doming": FunctionFeaturizer("doming", dominant_mode("doming"), navalue=None),
}

def augment_data(X, y):
    new_X = []
    new_y = []
    for macro_pos in MACROCYCLE_POSITIONS:
        for axial_pos in AXIAL_POSITIONS:
            for i in range(len(y)):
                equiv = X.iloc[i, :].loc[macro_pos + axial_pos + ["metal_radius"]].to_numpy()
                new_X.append(equiv)
                new_y.append(y.iloc[i, :])
    return pd.DataFrame(new_X), pd.DataFrame(new_y)


def run_fit(session, stype: str, task: str, featurizer: Featurizer, target: Featurizer, n_bootstraps: int=10, test_size: int=30, augment: bool=False):
    sids = utils.sids_by_type(session, stype)
    X = featurizer.featurize(session, sids)
    y = target.featurize(session, sids)
    if augment:
        X, y = augment_data(X, y)
    if task == "regression":
        model = RandomForestRegressor(n_estimators=1000)
    else:
        model = RandomForestClassifier(n_estimators=1000)
    return utils.train_model(task, model, X, y, n_bootstraps=n_bootstraps, test_size=test_size)

def analyze_bootstrap(df: pd.DataFrame, ci_alpha=None):
    """Method to calcualte average value of bootstrap experiments. optionally it adds CI information (in a separate column). if ci_alpha is none, returns a dataframe with average values"""
    avg = df.mean()
    if ci_alpha is not None:
        std = df.std()
        ci = stats.t(len(df) - 1).isf(ci_alpha / 2) * std / np.sqrt(len(df))
        avg = avg.to_frame()
        avg.columns = ["avg"]
        avg["ci"] = ci
        return avg
    else:
        avg = avg.to_frame()
        avg.columns = ["avg"]
        return avg


def fit_report(session, stype: str, task: str, display_metric: str, ci_alpha: float=0.05):
    data = []
    targets = REGRESSION_TARGETS if task == "regression" else CLASSIFICATION_TARGETS
    for feat in FEATURIZERS:
        ajr = {}
        for target in targets:
            print("RUNNING {} WITH {}".format(feat, target))
            _, df = run_fit(session, stype, task, FEATURIZERS[feat], targets[target])
            analyzed = analyze_bootstrap(df, ci_alpha)
            ajr[target + "_avg"] = analyzed.loc[display_metric, "avg"]
            if ci_alpha is not None:
                ajr[target + "_ci"] = analyzed.loc[display_metric, "ci"]
        data.append(ajr)
    return pd.DataFrame(data, index=FEATURIZERS.keys())


def plot_report(df: pd.DataFrame, ylabel: str, spacing: float=1, width: float=0.6):
    """display numbers in report as a bar plot"""
    avg_cols = [c for c in df.columns if "avg" in c]
    ci_cols = [c for c in df.columns if "ci" in c]
    x = np.linspace(0, len(avg_cols) * spacing + len(df) * len(avg_cols) * width, len(avg_cols))
    feat_positions = lambda center: np.linspace(center - len(df) * width / 2, center + len(df) * width / 2 - width, len(df))
    for i, feat in enumerate(df.index):
        pos = np.array([feat_positions(c)[i] for c in x])
        plt.bar(pos, df.loc[feat, avg_cols], width=width, align="edge", label=feat)
        plt.errorbar(pos + width / 2, df.loc[feat, avg_cols], df.loc[feat, ci_cols], linestyle="", ecolor="k")
    # fabulizing axis
    plt.xticks(x, [" ".join(c.split("_")[:-1]).title() for c in avg_cols])
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def influence_report(session, stype: str, task: str, featurizer: str, ci_alpha: float=0.05):
    data = pd.DataFrame()
    targets = REGRESSION_TARGETS if task == "regression" else CLASSIFICATION_TARGETS
    for target in targets:
        print("RUNNING {} WITH {}".format(featurizer, target))
        models, _ = run_fit(session, stype, task, FEATURIZERS[featurizer], targets[target], augment=True, n_bootstraps=5)
        df = pd.DataFrame([model.feature_importances_ for model in models], columns=FEATURIZERS[featurizer].feature_names)
        analyzed = analyze_bootstrap(df, ci_alpha)
        analyzed.columns = ["{}_{}".format(target, c) for c in analyzed.columns]
        for c in analyzed.columns:
            data[c] = analyzed[c]
        data.index = analyzed.index
    return data

if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    utils.define_pallet()
    engine = create_engine("sqlite:///{}".format("/home/shachar/repos/miscellaneous/CodStructures/main.db"))
    session = sessionmaker(bind=engine)()
    # df = fit_report(session, "porphyrin", "regression", "test_mae")
    # df.to_csv("results/regression_report.csv")
    # df = pd.read_csv("results/regression_report.csv", index_col=0)
    # plot_report(df, "Test set MAE")
    df = influence_report(session, "porphyrin", "regression", "vdw_distances")
    df.to_csv("results/influence_report.csv")