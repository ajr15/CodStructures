# module to contain all featurizers used for ML analysis
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from read_to_sql import StructureProperty

class Featurizer (ABC):

    def __init__(self, session, structure_ids, feature_names, **featurizer_kwargs):
        self.session = session
        self.structure_ids = structure_ids
        self.feature_names = feature_names
        self._featurized = self.featurize(session, structure_ids, **featurizer_kwargs)
        self._normalization_parameters = None

    @classmethod
    @abstractmethod
    def featurize(cls, session, structure_ids, **kwargs) -> np.array:
        pass

    def data(self):
        return self._featurized
    
    def normalize(self):
        features = self.data()
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        self._normalization_parameters = (means, stds)
        self._featurized = (features - means) / stds

    def reverse_normalize(self, data):
        if self._normalization_parameters is None:
            return data
        else:
            means, stds = self._normalization_parameters
            return data * stds + means
        
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(data=self.data(), columns=self.feature_names, index=self.structure_ids)
        
    def correlation_matrix_plot(self):
        df = self.to_dataframe()
        corr = df.corr() ** 2
        plt.matshow(corr, cmap="Greens")
        plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="left")
        plt.yticks(range(len(df.columns)), df.columns)
        plt.colorbar()

    def pca_plot(self, color_by):
        plt.figure()
        pca = PCA(n_components=2)
        vis = pca.fit_transform(self.data())
        if color_by is not None:
            if type(color_by[0]) is np.str_:
                for val in np.unique(color_by):
                    idxs = [i for i, c in enumerate(color_by) if c == val]
                    plt.scatter(vis[idxs][:, 0], vis[idxs][:, 1], label=val)
                plt.legend()
            else:
                plt.scatter(vis[:, 0], vis[:, 1], c=color_by, cmap="Greens")
                plt.colorbar()
        else:
            plt.scatter(vis[:, 0], vis[:, 1])
    
    def features_histograms(self):
        if len(self.features_names) > 8:
            nrows = int(np.ceil(len(self.features_names) / 2))
            ncols = 2
            axs_func = lambda axs, i: axs[i, 0] if i < nrows else axs[i - nrows, 1]
        else:
            nrows = len(self.features_names)
            ncols = 1
            axs_func = lambda axs, i: axs[i]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
        fig.tight_layout()
        features = self.data()
        for i, name in enumerate(self.features_names):
            ax = axs_func(axs, i)
            ax.hist(features[:, i], bins=30, color="green")
            ax.set_title(name)

    def correlation_plot(self, feat1_name, feat2_name):
        i1 = self.feature_names.index(feat1_name)
        i2 = self.feature_names.index(feat2_name)
        features = self.data()
        plt.scatter(features[:, i1], features[:, i2])


        
class StructurePropertyFeaturizer (Featurizer):

    def __init__(self, session, structure_ids, property_names, property_units):
        super().__init__(session, structure_ids, property_names, property_names=property_names, property_units=property_units)

    @classmethod
    def featurize(cls, session, structure_ids, property_names, property_units) -> np.array:
        res = []
        for sid in structure_ids:
            vec = []
            for pname, punits in zip(property_names, property_units):
                vec.append(cls.structure_property(session, sid, pname, punits))
            res.append(vec)
        return np.array(res)

    @staticmethod
    def structure_property(session, sid: int, property: str, units: str):
        q = session.query(StructureProperty.value).filter(StructureProperty.structure == sid).filter(StructureProperty.property == property)
        if units is not None:
            q = q.filter(StructureProperty.units == units)
        v = q.all()
        if len(v) == 0:
            raise ValueError("The property {} does not exists".format(property))
        else:
            return v[0][0]
