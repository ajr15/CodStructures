import pandas as pd
import os
import utils

def get_data(struct: str):
    """read the non-planarity CSV info"""
    path = os.path.join("results", struct + "_porphystruct.csv")
    ajr = pd.read_csv(path)
    return ajr.set_index("sid")

def self_correlation_plot(df: pd.DataFrame, title: str):
    plt.matshow(df.corr() ** 2, cmap="Greens")
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="left")
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title(title)
    plt.colorbar()

def correlation_analysis(struct: str):
    """Get correlation matrix for non-planarity modes"""
    df = get_data(struct)
    # clean data and separate to relative and absolute terms
    df = df[[c for c in df.columns if not c in ["sid", "structure"]]]
    rel_df = df[[c for c in df.columns if "%" in c or "non-planarity" in c]]
    abs_df = df[[c for c in df.columns if "%" not in c]]
    self_correlation_plot(rel_df, "Ratio Correlation")
    self_correlation_plot(abs_df, "Metric Correlation")

def distribution_analysis(struct: str):
    df = get_data(struct)
    plt.hist(df["non-planarity (exp.)"], color="k", bins=50)
    plt.title(struct.title() + " Total Non-Planarity")
    # calculate most prominent mode of non-planarity
    modes = df[[c for c in df.columns if "%" in c]]
    func = lambda row: row.apply(lambda x: 1 if x == row.max() else 0)
    modes = modes.apply(func, axis=1)
    modes = modes.mean(axis=0)
    modes = modes[modes > 0.01]
    if 1 - modes.sum() > 0.005:
        modes["Other"] = 1 - modes.sum()
    print(modes)
    plt.figure()
    plt.pie(modes, labels=[c.split()[0] for c in modes.index], autopct='%1.1f%%', startangle=90)


if __name__ == "__main__":
    import config
    import matplotlib.pyplot as plt
    args = utils.read_command_line_arguments("parse basic data from curated XYZ files. mostly around geometries.")
    # correlation_analysis(args.structure)
    distribution_analysis(args.structure)
    plt.show()