# script to use cone angle information to predict non-planarity
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, precision_score, recall_score
from sklearn.feature_selection import r_regression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from openbabel import openbabel as ob
from read_to_sql import Structure
from featurizers import StructurePropertyFeaturizer

def all_sids(session, stype):
    sids = session.query(Structure.id).filter(Structure.type == stype).\
                                                        distinct().all()
    return [x[0] for x in sids]


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

NON_PLANARITY_PROPERTIES = [
    "total out of plane (exp)",
    "saddling non planarity",
    "doming non planarity",
    "ruffling non planarity",
    "wavingx non planarity",
    "wavingy non planarity",
    "propellering non planarity"
]

NON_PLANARITY_UNITS = "A"

if __name__ == "__main__":
    engine = create_engine("sqlite:///{}".format("./main.db"))
    session = sessionmaker(bind=engine)()
    sids = all_sids(session, "porphyrin")
    x_features = StructurePropertyFeaturizer(session, sids, NON_PLANARITY_PROPERTIES, ["A"] + [NON_PLANARITY_UNITS for _ in NON_PLANARITY_PROPERTIES[1:]])
    y_features = StructurePropertyFeaturizer(session, sids, ["outer_circuit en"], [None])
    X = x_features.data()
    y = y_features.data()
    plt.figure()
    plt.scatter(X[:, 0], y)
    plt.title("Total out of plane vs. HOMA")
    plt.figure()
    plt.hist(y)
    plt.title("HOMA Distribution")
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
    model = RandomForestRegressor(n_estimators=500)
    model.fit(xtrain, ytrain)
    res = {}
    res.update(regression_fit_metrics(model, xtrain, ytrain, "train_"))
    res.update(regression_fit_metrics(model, xtest, ytest, "test_"))
    # make fit plots
    plt.figure()
    plt.scatter(model.predict(xtrain), ytrain, c="#5AA2AE")
    plt.title("Train")
    plt.figure()
    plt.scatter(model.predict(xtest), ytest, c="#5AA2AE")
    plt.title("Test")
    res.update({name: value for name, value in zip(x_features.feature_names, model.feature_importances_)})
    df = pd.DataFrame([res])
    print(df)
    df.to_csv("results/rf_fit_results.csv")
    plt.show()