import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA


class Dataset:

    def __init__(self, path):
        self.encoders = None
        self.scaler = None
        self.norm_df = None
        self.pca = None
        data = arff.loadarff(path)
        self.df = pd.DataFrame(data[0])
        self.df_backup = self.df.copy()
        self.encode_columns()

    def encode_columns(self):
        object_df = self.df.select_dtypes(include=['object'])
        self.encoders = {}
        for col in object_df.columns:
            ord_enc = OrdinalEncoder()
            self.df[col] = ord_enc.fit_transform(object_df[[col]])
            self.encoders[col] = ord_enc

    def inverse_encoder(self, df):
        for key, value in self.encoders.items():
            df[key] = value.inverse_transform(df[key].values.reshape(-1, 1))
        return df

    def norm(self, method='Standard'):
        if method == 'Standard':
            self.scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise Exception("Not defined normalization method")
        class_col = self.df.iloc[:, -1]
        object_df = self.df.select_dtypes(exclude=['object'])
        norm_values = self.scaler.fit_transform(object_df.values[:, :-1])
        object_df = pd.DataFrame(norm_values, columns=object_df.columns.values[:-1])
        self.norm_df = object_df
        self.norm_df[self.df.columns[-1]] = class_col
        return object_df
    
    def learn_embeddings(self):
        self.pca = PCA(n_components=2, svd_solver='full')
        self.pca.fit(self.norm_df.values)
        self.pca_df = self.pca.transform(self.norm_df.values)
        return self.pca_df
    
    def transform_data_pca(self, data):
        if self.pca == None:
            return None
        return self.pca.transform(data)
