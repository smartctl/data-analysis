# general libraries
import pprint
# common data manipulation libraries
import pandas as pd
import numpy as np

# loading common ML libraries and setting defaults
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    StandardScaler
)

# plotting library
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

class CustomML:
    def __init__(self, X_data: pd.DataFrame, random_state=42, split_size=0.25) -> None:
        self.X_data=X_data.copy().drop(['cluster_magic_split'], axis=1)                     
        self.encode_y()
        # to store per-custer data
        self.X_cluster = self.define_X_cluster()
        self.random_state=random_state
        self.split_size=split_size

        self.split_xy()
        self.split_xy_by_cluster_type()

        print(f"Loaded dataset with shape {self.X_data.shape}")


    def encode_y(self):
        """
        Encodes y_label in numeric format
            0 = green
            1 = red
            2 = yellow
        """
        self.X_data.loc[self.X_data['y_label'] == "green",  ['y_label']]=0
        self.X_data.loc[self.X_data['y_label'] == "red",    ['y_label']]=1
        self.X_data.loc[self.X_data['y_label'] == "yellow", ['y_label']]=2

    def define_X_cluster(self):
        X_cluster = {
            'compact':{
                'X_full':None,
                'X':None, 
                'y': None, 
                'X_train': None, 
                'y_train': None, 
                'X_test': None, 
                'y_test': None
            },
            'mno':{
                'X_full':None,
                'X':None, 
                'y': None, 
                'X_train': None, 
                'y_train': None, 
                'X_test': None, 
                'y_test': None
            }
        } 
        return X_cluster

    def split_by_cluster_type(self):
        """
        Create an dataset per cluster type
        """
        self.X_cluster['compact']['X_full'] = self.X_data[self.X_data['source'] == 'compact']
        self.X_cluster['mno']['X_full']     = self.X_data[self.X_data['source'] == 'mno']

        print(f"Compact cluster {self.X_cluster['compact']['X_full'].shape}"+
              f"\nMultinode (mno) {self.X_cluster['mno']['X_full'].shape}")

    def split_xy_by_cluster_type(self):
        """
        Create an X and y per cluster type and the corresponding train_test_split
        """
        if self.X_cluster['compact']['X_full'] is None:
            self.split_by_cluster_type()

        self.X_cluster['compact']['X'] = self.X_cluster['compact']['X_full'].drop(['source','y_label'], axis=1)
        self.X_cluster['compact']['y'] = self.X_cluster['compact']['X_full']['y_label']

        self.X_cluster['mno']['X'] = self.X_cluster['mno']['X_full'].drop(['source','y_label'], axis=1)
        self.X_cluster['mno']['y'] = self.X_cluster['mno']['X_full']['y_label']

        for cluster in self.X_cluster.keys():
            X_train, X_test, y_train, y_test = train_test_split(self.X_cluster[cluster]['X'], 
                                                                self.X_cluster[cluster]['y'], 
                                                                test_size=self.split_size, 
                                                                random_state=self.random_state)
            self.X_cluster[cluster]['X_train'] = X_train
            self.X_cluster[cluster]['X_test']  = X_test
            self.X_cluster[cluster]['y_train'] = y_train
            self.X_cluster[cluster]['y_test']  = y_test


    def split_xy(self):
        """
        Create a X_full and y_full
        """
        self.X = self.X_data.drop(['source','y_label'], axis=1)
        self.y = self.X_data['y_label']

    def get_dataset(self, dataset_type="full"):
        """
        return dataset_type

        dataset_type
            "full"    uses the full dataset X_data
            "compact" uses the X_compact dataset
            "mno"     uses the X_mno dataset
        """
        match dataset_type:
            case "full":
                df=self.X_data.copy()
            case "compact":
                df=self.X_cluster['compact']['X'].copy()
            case "mno":
                df=self.X_cluster['compact']['mno'].copy()
            case _:
                print(f"Invalid dataset_type")
                return None
        return df
    
    def get_xy_by_cluster_type(self):
        if self.X_cluster['compact']['X'] is None:
            self.split_xy_by_cluster_type()
        return self.X_cluster

    def get_xy_logreg(self, dataset_type="full"):
        """
        return X, y wth y_label encoded for LogisticRegression

        dataset_type
            "full"    uses the full dataset X_data
            "compact" uses the X_compact dataset
            "mno"     uses the X_mno dataset
        """
        df=self.get_dataset(dataset_type)

        # Default datasets encoding: 0 = green, 1 = red, 2 = yellow
        # Chaning any yellow to 'red' encoded as 1
        df.loc[df['y_label'] == 2, ['y_label']]=1
        
        print(f"{df['y_label'].value_counts()}")

        # Get our full logreg X and y
        X_logreg = df.drop(['source','y_label'], axis=1)
        y_logreg = df['y_label'].astype('boolean')

        return X_logreg,y_logreg

    def feature_scaling_per_type(self, df, scaler_uint32='standard', scaler_float64='standard'):
        """
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

        valid scalers for this function are:
            'standard'  | StandardScaler
            'maxabs'    | MaxAbsScaler
            'minmax'    | MinMaxScaler
            'power'     | PowerTransformer
        """
        #print(f"Scaling dataset. Make sure to have the train and test split BEFORE the scaling")

        # pipeline to transform uint32 features
        # These features represent an absolute value (e.g. num of Pods)
        pipe_uint32 = { 
            'standard' : Pipeline([
                ('scaler', StandardScaler())
                ]), 
            'maxabs' : Pipeline([
                ('maxabs', MaxAbsScaler())
                ]),
            'minmax' : Pipeline([
                ('minmax', MinMaxScaler(feature_range=(0,1)))
                ]),
            'power' : Pipeline([
                ('power', PowerTransformer(method='yeo-johnson'))
                ])
        }

        # pipeline to transform float64 features
        # These features represent a percentages, transaction rates
        pipe_float64 = { 
            'standard' : Pipeline([
                ('scaler', StandardScaler())
                ]), 
            'maxabs' : Pipeline([
                ('maxabs', MaxAbsScaler())
                ]),
            'minmax' : Pipeline([
                ('minmax', MinMaxScaler(feature_range=(0,1)))
                ]),
            'power' : Pipeline([
                ('power', PowerTransformer(method='yeo-johnson'))
                ])
        }

        ct = ColumnTransformer([
                (
                    'scaler_uint32', pipe_uint32[scaler_uint32],
                    df.select_dtypes(include=['uint32']).columns.to_list()
                ),
                (
                    'scaler_float64', pipe_float64[scaler_float64], 
                    df.select_dtypes(include=['float64']).columns.to_list()
                )
            ])

        return ct

    def pipeline_transformer_logreg(self, X_train, y_train, X_test, y_test, scaler_pairs=[('standard','standard')]):
        """
        """
        score_pair={}
       # LogisticRegression instance with settings
        logreg = LogisticRegression(random_state=self.random_state, penalty='l2',
                                    solver='newton-cholesky',
                                    class_weight='balanced',
                                    max_iter=3000) 
        
        for scaler1,scaler2 in scaler_pairs:
            ct  = self.feature_scaling_per_type(X_train, scaler1, scaler2)
            pipeline = Pipeline([
                            ('column_transformer', ct),
                            ('logistic_regression', logreg)
            ])
            pipeline.fit(X_train, y_train)
            score_pair[scaler1+","+scaler2]=pipeline.score(X_test, y_test)
        
        pprint.pprint(score_pair)

