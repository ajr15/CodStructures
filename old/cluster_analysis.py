from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm
import pandas as pd
from featurizers import StructurePropertyFeaturizer
from read_to_sql import Structure

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

def all_sids(session, stype):
    sids = session.query(Structure.id).filter(Structure.type == stype).\
                                                        distinct().all()
    return [x[0] for x in sids]

def entropy(x, mean, std):
    """Calculate the entropy of a given variable, assuming it has a normal distribution"""
    return np.mean(- np.log(norm(mean, std).pdf(x)))

if __name__ == "__main__":
    engine = create_engine("sqlite:///{}".format("./main.db"))
    session = sessionmaker(bind=engine)()
    sids = all_sids(session, "porphyrin")
    x_features = StructurePropertyFeaturizer(session, sids, NON_PLANARITY_PROPERTIES, ["A"] + [NON_PLANARITY_UNITS for _ in NON_PLANARITY_PROPERTIES[1:]])
    x_features.normalize()
    y_features = StructurePropertyFeaturizer(session, sids, ["structure homa"], [None])
    X = x_features.data()
    y = y_features.data()
    ymean = np.mean(y)
    ystd = np.std(y)    
    base_entropy = entropy(y, ymean, ystd)
    data = []
    for n in range(2, 10):
        print(n)
        model = GaussianMixture(n_components=n)
        model.fit(X)
        clusters = model.predict(X)
        new_entropy = 0
        for c in np.unique(clusters):
            idxs = [i for i, cl in enumerate(clusters) if cl == c]
            new_entropy += entropy(y[idxs], ymean, ystd)
        information_gain = new_entropy - base_entropy
        data.append({"N": n, "gain": information_gain})
    df = pd.DataFrame(data)
    df.to_csv('information_gain.csv')
    print(df)


