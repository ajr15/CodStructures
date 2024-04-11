# script to present basic description of the data in hand
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from openbabel import openbabel as ob
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker
from read_to_sql import Structure, Substituent, StructureProperty, SubstituentProperty
from utils import mol_from_smiles

def all_sids(session):
    sids = session.query(Structure.id).all()
    return [x[0] for x in sids]

def coordination_number(session, sid: str):
    n_axials = session.query(Substituent).filter(Substituent.structure == sid).filter(Substituent.position == "axial").count()
    return 4 + n_axials

def structure_type(session, sid: str):
    return session.query(Structure.type).filter(Structure.id == sid).all()[0][0]

def central_metal(session, sid: str):
    smiles = session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "metal").all()[0][0]
    return smiles[1:-1]

def metal_charge(session, sid: str):
    return session.query(SubstituentProperty.value).filter(SubstituentProperty.structure == sid).filter(SubstituentProperty.property == "charge").all()[0][0]

def number_of_atoms(session, sid: str):
    smiles = session.query(Structure.smiles).filter(Structure.id == sid).all()[0][0]
    mol = mol_from_smiles(smiles)
    return mol.NumAtoms()

def molar_mass(session, sid: str):
    smiles = session.query(Structure.smiles).filter(Structure.id == sid).all()[0][0]
    mol = mol_from_smiles(smiles)
    return mol.GetMolWt()

def build_dataframe(property_dict: dict, update: bool=False):
    engine = create_engine("sqlite:///{}".format("./main.db"))
    session = sessionmaker(bind=engine)()
    data = []
    for sid in all_sids(session):
        d = {"sid": sid}
        for name, func in property_dict.items():
            if update:
                d.update(func(session, sid))
            else:
                d[name] = func(session, sid)
        data.append(d)
    return pd.DataFrame(data)

def structure_details_csv():
    """Save all structure details information into a CSV file"""
    properties = {
        "structure_type": structure_type,
        "coordination_number": coordination_number,
        "central_metal": central_metal,
        "metal_charge": metal_charge,
        "number_of_atoms": number_of_atoms,
        "molar_mass": molar_mass
    }
    ajr = build_dataframe(properties)
    ajr.to_csv("results/structure_details.csv")

def total_out_of_plane(session, sid):
    val = session.query(StructureProperty.value).\
        filter(StructureProperty.structure == sid).\
        filter(StructureProperty.property == "total out of plane (exp)").all()[0][0]
    return val

def n_hydrogen_substitutions(session, sid, position):
    val = session.query(Substituent).filter(Substituent.structure == sid).filter(Substituent.position == position).filter(Substituent.substituent == "[*H]").count()
    return val


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

def non_planarity_csv():
    properties = {
        1: lambda session, sid: {"structure_type": structure_type(session, sid)},
        2: lambda session, sid: {"total non planarity": total_out_of_plane(session, sid)},
        3: lambda session, sid: all_non_planarity(session, sid, "A"),
        4: lambda session, sid: all_non_planarity(session, sid, "%")
    }
    ajr = build_dataframe(properties, update=True)
    ajr.to_csv("results/non_planarity.csv")
    return ajr

def total_non_planarity_hist(df):
    plt.rcParams["font.size"] = 16
    xmax = df["total non planarity"].max()
    fig, axs = plt.subplots(ncols=2, figsize=(13, 6))
    axs[0].hist(df[df["structure_type"] == "porphyrin"]["total non planarity"], color="#5AA2AE", bins=30)
    axs[0].set_title("Porphyrins")
    axs[0].set_xlim(0, xmax)
    axs[1].hist(df[df["structure_type"] == "corrole"]["total non planarity"], color="#5AA2AE", bins=10)
    axs[1].set_title("Corroles")
    axs[1].set_xlim(0, xmax)

def metal_radius(session, sid: int):
    """Get the VDW radius of the metal center"""
    metal = session.query(Substituent.substituent).filter(Substituent.structure == sid).filter(Substituent.position == "metal").all()[0][0]
    metal = mol_from_smiles(metal).GetAtom(1)
    return ob.GetVdwRad(metal.GetAtomicNum())

def non_planarity_modes_hist(df, units):
    plt.rcParams["font.size"] = 12
    props = ["doming", "saddling", "ruffling", "wavingx", "wavingy", "propellering"]
    fig, axs = plt.subplots(nrows=len(props), ncols=2, figsize=(10, 10))
    # plt.tight_layout()
    xmin = df[[c for c in df.columns if "({})".format(units) in c]].min().min()
    xmax = df[[c for c in df.columns if "({})".format(units) in c]].max().max()
    for i, prop in enumerate(props):
        axs[i, 0].hist(df[df["structure_type"] == "porphyrin"]["{} ({})".format(prop, units)], color="#5AA2AE", bins=30)
        axs[i, 0].set_ylabel(prop.title())
        axs[i, 0].set_xlim(xmin, xmax)
        axs[i, 0].set_xticks([])
        axs[i, 1].hist(df[df["structure_type"] == "corrole"]["{} ({})".format(prop, units)], color="#5AA2AE", bins=10)
        axs[i, 1].set_xlim(xmin, xmax)
        axs[i, 1].set_xticks([])
    axs[-1, 0].set_xticks(np.linspace(xmin, xmax, 4).round())
    axs[-1, 1].set_xticks(np.linspace(xmin, xmax, 4).round())

def non_planarity_plots():
    plt.style.use("seaborn-v0_8-deep")
    df = non_planarity_csv()
    # plotting non-planarity histograms
    # total_non_planarity_hist(df)
    # plotting individual values
    non_planarity_modes_hist(df, "A")
    non_planarity_modes_hist(df, "%")
    plt.show()


def make_violin_plot(ax, df: pd.DataFrame, xcol, ycol):
    values = sorted(df[xcol].unique())
    plot_vals = []
    for v in values:
        plot_vals.append(df[df[xcol] == v][ycol])
    ax.violinplot(plot_vals, positions=values, showmeans=True)
    ax.set_xlabel(xcol.replace("_", " ").title())
    ax.set_ylabel(ycol.replace("_", " ").title())


def non_planarity_correlations():
    properties = {
        "structure_type": structure_type,
        "coordination_number": coordination_number,
        "n_beta_hydrogens": lambda session, sid: n_hydrogen_substitutions(session, sid, "beta"),
        "n_meso_hydrogens": lambda session, sid: n_hydrogen_substitutions(session, sid, "meso"),
        "metal_radius": metal_radius,
        "metal_charge": metal_charge,
        "total_non_planarity": total_out_of_plane
    }
    df = build_dataframe(properties)
    df.to_csv("results/hydrogen_substitutions.csv")
    corroles_df = df[df["structure_type"] == "corrole"]
    porphyrins_df = df[df["structure_type"] == "porphyrin"]
    porphyrins_df["total_h_substituents"] = porphyrins_df["n_beta_hydrogens"] + porphyrins_df["n_meso_hydrogens"]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    make_violin_plot(axs[0, 0], corroles_df, "n_beta_hydrogens", "total_non_planarity")
    make_violin_plot(axs[1, 0], corroles_df, "coordination_number", "total_non_planarity")
    make_violin_plot(axs[0, 1], porphyrins_df, "total_h_substituents", "total_non_planarity")
    make_violin_plot(axs[1, 1], porphyrins_df, "coordination_number", "total_non_planarity")
    fig, axs = plt.subplots(ncols=2)
    make_violin_plot(axs[0], corroles_df, "metal_charge", "total_non_planarity")
    make_violin_plot(axs[1], porphyrins_df, "metal_charge", "total_non_planarity")
    plt.show()

if __name__ == "__main__":
    non_planarity_correlations()