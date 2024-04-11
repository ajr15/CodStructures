# script to use cone angle information to predict non-planarity
import pandas as pd
from functools import reduce
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import r_regression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from openbabel import openbabel as ob
from read_to_sql import SubstituentProperty, StructureProperty, Substituent, Structure
from utils import mol_from_smiles


def get_structure_property(session, sid: int, property: str, units: str):
    v = session.query(StructureProperty.value).filter(StructureProperty.structure == sid).filter(StructureProperty.property == property).filter(StructureProperty.units == units).all()
    if len(v) == 0:
        raise ValueError("The property {} does not exists".format(property))
    else:
        return v[0][0]
    
def all_non_planarity(session, sid, units):
    structprops = session.query(StructureProperty).\
                filter(StructureProperty.structure == sid).\
                filter(StructureProperty.property.contains("non planarity")).\
                filter(StructureProperty.units == units).all()
    ajr = {}
    for structprop in structprops:
        key = "{} ({})".format(structprop.property[:-14], units)
        ajr[key] = structprop.value
    return ajr

def get_dominant_mod(session, sid, th):
    structprops = session.query(StructureProperty).\
                filter(StructureProperty.structure == sid).\
                filter(StructureProperty.property.contains("non planarity")).\
                filter(StructureProperty.units == "%").all()
    max_mode = reduce(lambda x, y: x if x.value > y.value else y, structprops)
    if max_mode.value >= th:
        return max_mode.property.split()[0]
    else:
        return "undefined"



def is_dominant(session, sid, mod, th):
    dominant = get_dominant_mod(session, sid, th)
    return mod == dominant

DOMINANCE_TH = 0.3


PROPERTIES = {
    "total out of plane": lambda session, sid: get_structure_property(session, sid, "total out of plane (exp)", "A"),
    "saddlingA": lambda session, sid: get_structure_property(session, sid, "saddling non planarity", "A"),
    "domingA": lambda session, sid: get_structure_property(session, sid, "doming non planarity", "A"),
    "rufflingA": lambda session, sid: get_structure_property(session, sid, "ruffling non planarity", "A"),
    "wavingxA": lambda session, sid: get_structure_property(session, sid, "wavingx non planarity", "A"),
    "wavingyA": lambda session, sid: get_structure_property(session, sid, "wavingy non planarity", "A"),
    "propelleringA": lambda session, sid: get_structure_property(session, sid, "propellering non planarity", "A"),
    "saddling%": lambda session, sid: get_structure_property(session, sid, "saddling non planarity", "%"),
    "doming%": lambda session, sid: get_structure_property(session, sid, "doming non planarity", "%"),
    "ruffling%": lambda session, sid: get_structure_property(session, sid, "ruffling non planarity", "%"),
    "wavingx%": lambda session, sid: get_structure_property(session, sid, "wavingx non planarity", "%"),
    "wavingy%": lambda session, sid: get_structure_property(session, sid, "wavingy non planarity", "%"),
    "propellering%": lambda session, sid: get_structure_property(session, sid, "propellering non planarity", "%"),
    "saddling": lambda session, sid: is_dominant(session, sid, "saddling", DOMINANCE_TH),
    "doming": lambda session, sid: is_dominant(session, sid, "doming", DOMINANCE_TH),
    "ruffling": lambda session, sid: is_dominant(session, sid, "ruffling", DOMINANCE_TH),
    "wavingx": lambda session, sid: is_dominant(session, sid, "wavingx", DOMINANCE_TH),
    "wavingy": lambda session, sid: is_dominant(session, sid, "wavingy", DOMINANCE_TH),
    "propellering": lambda session, sid: is_dominant(session, sid, "propellering", DOMINANCE_TH),
    "dominant_mode": lambda session, sid: get_dominant_mod(session, sid, DOMINANCE_TH),
}


RAW_CONE_ANGLE_FEATURES = [
    "axial1", 
    "axial2", 
    "beta1", 
    "beta2", 
    "beta3", 
    "beta4", 
    "beta5", 
    "beta6", 
    "beta7", 
    "beta8",
    "meso1",
    "meso2",
    "meso3",
    "meso4",
    "metal"
]

AGGREGATED_ANGLE_FEATURES = [
    "axial1",
    "axial2",
    "metal",
    "angle_sum_avg",
    "meso_angle_avg",
    "meso_angle_freq",
    "beta_angle_avg",
    "beta_angle_freq",
]

BETA_METAL_COUPLES = {
    "corrole": [
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (3, 6),
        (3, 7)
    ],
    "porphyrin": [
        (1, 1),
        (1, 8),
        (2, 2),
        (2, 3),
        (3, 4),
        (3, 5),
        (4, 6),
        (4, 7)
    ]
}

class FeaturizationError (Exception):
    pass

def coning_angles_vec(session, sid: int):
    """Get the vector of coning angles for a given structure"""
    properties = session.query(SubstituentProperty).filter(SubstituentProperty.property == "cone angle").\
                                             filter(SubstituentProperty.structure == sid).\
                                             order_by(SubstituentProperty.position, SubstituentProperty.position_index).\
                                             all()
    # adding all angles to vector
    vec = []
    for p in properties:
        vec.append(p.value)
    # pad for 2 axial ligands
    naxial = len([p for p in properties if p.position == "axial"])
    vec = [0 for _ in range(2 - naxial)] + vec
    # pad for 4 meso ligands
    nmeso = len([p for p in properties if p.position == "meso"])
    vec = vec + [0 for _ in range(4 - nmeso)]
    # add metal radius
    vec += metal_radius(session, sid)
    if len(vec) != 15:
        raise FeaturizationError("Length of feature for sid={} is bad".format(sid))
    return vec

def calcualte_angle_sums(stype, angle_dict):
    ajr = []
    for meso_idx, beta_idx in BETA_METAL_COUPLES[stype]:
        ajr.append(angle_dict["meso" + str(meso_idx)] + angle_dict["beta" + str(meso_idx)]) 
    return ajr


def aggregated_features_vec(session, sid: int):
    stype = session.query(Structure.type).filter(Structure.id == sid).all()[0][0]
    angle_dict = {k: val for k, val in zip(RAW_CONE_ANGLE_FEATURES, coning_angles_vec(session, sid))}
    angle_sums = calcualte_angle_sums(stype, angle_dict)
    meso_angles = [angle_dict["meso" + str(i + 1)] for i in range(4)]
    beta_angles = [angle_dict["beta" + str(i + 1)] for i in range(8)]
    return [
        angle_dict["axial1"],
        angle_dict["axial2"],
        angle_dict["metal"],
        np.mean(angle_sums),
        np.mean(meso_angles),
        np.bincount(meso_angles).argmax(),
        np.mean(beta_angles),
        np.bincount(beta_angles).argmax(),
    ]

def metal_radius(session, sid: int):
    """Get the VDW radius of the metal center"""
    metal = session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "metal").all()[0][0]
    metal = mol_from_smiles(metal).GetAtom(1)
    return [ob.GetVdwRad(metal.GetAtomicNum())]


def all_sids(session):
    sids = session.query(SubstituentProperty.structure).filter(SubstituentProperty.property == "cone angle").\
                                                        distinct().all()
    return [x[0] for x in sids]

def build_features_vectors(session, featurization_func):
    sids = all_sids(session)
    X = []
    ajr = []
    for sid in sids:
        try:
            xvec = featurization_func(session, sid)
            X.append(xvec)
            ajr.append(sid)
        except FeaturizationError:
            print("ERRORS WITH", sid, "PLEASE CHECK IT OUT, FOR NOW WE IGNORE")
    return np.array(X), ajr

def build_property_vectors(session, sids, prop):
    y = []
    for sid in sids:
        y.append(PROPERTIES[prop](session, sid))
    return np.array(y)

def pca_visualize(X, y):
    plt.figure()
    pca = PCA(n_components=2)
    vis = pca.fit_transform(X)
    if type(y[0]) is np.str_:
        for val in np.unique(y):
            idxs = [i for i, y in enumerate(y) if y == val]
            plt.scatter(vis[idxs][:, 0], vis[idxs][:, 1], label=val)
        plt.legend()
    else:
        plt.scatter(vis[:, 0], vis[:, 1], c=y, cmap="Greens")
        plt.colorbar()

def regression_fit_metrics(model, X, y, prefix=""):
    pred = model.predict(X)
    return {
        prefix + "mae": mean_absolute_error(y, pred),
        prefix + "mare": mean_absolute_percentage_error(y, pred),
        prefix + "r2": r_regression(y.reshape(-1, 1), pred)[0]**2
    }

def classification_fit_metrics(model, X, y, prefix=""):
    pred = model.predict(X)
    return {
        prefix + "accuracy": accuracy_score(y, pred),
        prefix + "precision": precision_score(y, pred),
        prefix + "recall": recall_score(y, pred)
    }

def features_histograms(features_vector, features_names):
    if len(features_names) > 8:
        nrows = int(np.ceil(len(features_names) / 2))
        ncols = 2
        axs_func = lambda axs, i: axs[i, 0] if i < nrows else axs[i - nrows, 1]
    else:
        nrows = len(features_names)
        ncols = 1
        axs_func = lambda axs, i: axs[i]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    fig.tight_layout()
    for i, name in enumerate(features_names):
        ax = axs_func(axs, i)
        ax.hist(features_vector[:, i], bins=30, color="green")
        ax.set_title(name)


def explore_features(featurization_func, feature_names):
    engine = create_engine("sqlite:///{}".format("./main.db"))
    session = sessionmaker(bind=engine)()
    X, valid_sids = build_features_vectors(session, featurization_func)
    # plotting histograms
    features_histograms(X, feature_names)
    # plotting correlation matrix
    df = pd.DataFrame(data=X, columns=feature_names).corr()
    plt.matshow(df ** 2, cmap="Greens")
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="left")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
    # making PCA visualization with total non planarity
    total_non_planarity = build_property_vectors(session, valid_sids, "total out of plane")
    pca_visualize(X, total_non_planarity)
    dominant_mode = build_property_vectors(session, valid_sids, "dominant_mode")
    pca_visualize(X, dominant_mode)
    plt.show()

if __name__ == "__main__":
    engine = create_engine("sqlite:///{}".format("./main.db"))
    session = sessionmaker(bind=engine)()
    X, valid_sids = build_features_vectors(session, aggregated_features_vec)
    fit_results = []
    for prop in ["saddling", "doming", "ruffling", "wavingx", "wavingy", "propellering"]:
        print(prop)
        y = build_property_vectors(session, valid_sids, prop)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=50, random_state=0)
        model = RandomForestClassifier(n_estimators=500)
        model.fit(xtrain, ytrain)
        res = {"property": prop}
        res.update(classification_fit_metrics(model, xtrain, ytrain, "train_"))
        res.update(classification_fit_metrics(model, xtest, ytest, "test_"))
        res.update({name: value for name, value in zip(AGGREGATED_ANGLE_FEATURES, model.feature_importances_)})
        fit_results.append(res)
    # run with total non planarity
    prop = "total out of plane"
    y = build_property_vectors(session, valid_sids, prop)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=50, random_state=0)
    model = RandomForestRegressor(n_estimators=500)
    model.fit(xtrain, ytrain)
    res = {"property": prop}
    res.update(regression_fit_metrics(model, xtrain, ytrain, "train_"))
    res.update(regression_fit_metrics(model, xtest, ytest, "test_"))
    # make fit plots
    plt.scatter(model.predict(xtrain), ytrain, c="#5AA2AE")
    plt.title("Train")
    plt.figure()
    plt.scatter(model.predict(xtest), ytest, c="#5AA2AE")
    plt.title("Test")
    res.update({name: value for name, value in zip(AGGREGATED_ANGLE_FEATURES, model.feature_importances_)})
    fit_results.append(res)

    df = pd.DataFrame(fit_results)
    print(df)
    df.to_csv("results/rf_fit_results.csv")
    plt.show()