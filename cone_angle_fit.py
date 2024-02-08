# script to use cone angle information to predict non-planarity
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from openbabel import openbabel as ob
from read_to_sql import SubstituentProperty, StructureProperty, Substituent
from utils import mol_from_smiles

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
    return vec

def metal_radius(session, sid: int):
    """Get the VDW radius of the metal center"""
    metal = session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "metal").all()[0][0]
    metal = mol_from_smiles(metal).GetAtom(1)
    return [ob.GetVdwRad(metal.GetAtomicNum())]

def get_structure_property(session, sid: int, property: str, units: str):
    return session.query(StructureProperty.value).filter(StructureProperty.structure == sid).filter(StructureProperty.property == property).filter(StructureProperty.units == units).all()[0][0]

def all_sids(session):
    sids = session.query(SubstituentProperty.structure).filter(SubstituentProperty.property == "cone angle").\
                                                        distinct().all()
    return [x[0] for x in sids]

def make_data(session, property: str, units: str):
    sids = all_sids(session)
    X = []
    y = []
    for sid in sids:
        xvec = coning_angles_vec(session, sid)
        if len(xvec) == 15:
            X.append(xvec)
            y.append(get_structure_property(session, sid, property, units))
        else:
            print("ERRORS WITH", sid, "PLEASE CHECK IT OUT, FOR NOW WE IGNORE")
    return np.array(X), np.array(y)

def pca_visualize(X, y):
    pca = PCA(n_components=2)
    vis = pca.fit_transform(X)
    plt.scatter(vis[:, 0], vis[:, 1], c=y, cmap="Greens")
    plt.colorbar()

if __name__ == "__main__":
    engine = create_engine("sqlite:///{}".format("./main.db"))
    session = sessionmaker(bind=engine)()
    X, y = make_data(session, "total out of plane (exp)", "A")
    pca_visualize(X, y)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=50)
    model = RandomForestRegressor()
    model.fit(xtrain, ytrain)
    plt.figure()
    plt.plot(model.predict(xtrain), ytrain, "ko")
    plt.title("Train")
    plt.figure()
    plt.plot(model.predict(xtest), ytest, "ko")
    plt.title("Test")
    plt.show()
