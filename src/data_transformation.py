import pandas as pd
from datetime import datetime
import os, re, time, inspect
from .config import Logger

class DataTransformation:
    """
    Utility functions for data transofmration
    """
    def __init__(self, y_label_yellow=[], y_label_red=[], y_label_red_fatal=[], logger=None):
        self.logger = logger if logger else Logger(show_message=False).logger
        self.logger.debug(f"[{inspect.stack()[0][3]}] Initializing DataTransformation class.")
        self.y_label_yellow = y_label_yellow
        self.y_label_red = y_label_red
        self.y_label_red_fatal = y_label_red_fatal
        self.logger.debug(f"[{inspect.stack()[0][3]}] DataTransformation initialization completed.")

    def impute_y_label(self, df, label_weight=[0.25,0.10,0.00]):
        """
        Create a final y_label column with 'green', 'yellow' or 'red'

        label_weight=[yellow, red, red_fatal]
                    is the percentage (0.0 <= w <= 1.0) for the treshold for applying the y_label color.
                    When the weight is 0% it meaans that anything > 0 will trigger the color
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting to impute y_label.")


        for w in label_weight:
            if not 0 <= w <= 1:
                self.logger.error(f"[{inspect.stack()[0][3]}] y_label weight must be (0 <= w <= 1). Usign default label_weight=[0.25,0.10,0.00]")
                label_weight=[0.25,0.10,0.010]
                break

        total_yy_labels = df.filter(regex="yy").shape[1] # total number of yy features
        df['y_label']='green' # by default assume everything is green

        # initialize columns calculating the number of labels per type (yellow, red, red_fatal)
        self.logger.warn(f"[{inspect.stack()[0][3]}] Known issue. Requires optimization for performance.")
        logical_or="|"
        df['total_y_label_yellow']=df.filter(regex=logical_or.join(self.y_label_yellow)).sum(axis=1).astype('uint32')
        df['total_y_label_red']=df.filter(regex=logical_or.join(self.y_label_red)).sum(axis=1).astype('uint32')
        df['total_y_label_red_fatal']=df.filter(regex=logical_or.join(self.y_label_red_fatal)).sum(axis=1).astype('uint32')

        # formula for yellow state
        df.loc[
            (
                ( df['total_y_label_yellow'] 
                 + (df['total_y_label_red'] * 2)           # Each red counts double
                 ) > round(total_yy_labels * label_weight[0])
            ), 
            ['y_label']]='yellow'
        
        # if has red above thresshold assign color red
        df.loc[(df['total_y_label_red'] > round(total_yy_labels * label_weight[1])), ['y_label']]='red'

        # if has any red_fatal then y_label=red
        df.loc[(df['total_y_label_red_fatal'] > round(total_yy_labels * label_weight[2])), ['y_label']]='red'

        # Additional overrides for y_labels based on qty of active control planes
        df.loc[
            (df['y_label'] != 'red') & 
            (df['total_qty_control_plane'] < 2),
            'y_label'] = "red"

        df.loc[
            (df['y_label'] != 'red') & 
            (df['total_qty_control_plane'] == 2),
            'y_label'] = "yellow"
        
        self.logger.debug(f"[{inspect.stack()[0][3]}] Ending imputing y_label from {total_yy_labels} columns.")
        return df

    def y_by_group(self, df):
        """
        Print aggregations of y_label
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting displaying y_label aggregation.")

        print(f"----\nBy y_label:\n {df.groupby(['y_label'])['y_label'].count()}\n")
        print(f"----\nBy source by y_label:\n {df.groupby(['source','y_label'])['source'].count()}")

        self.logger.debug(f"[{inspect.stack()[0][3]}] Ending displaying y_label aggregation.")

    def normalize_sources(self, df):
        """
        Consolidate clusters into two groups: compact and mno
            compact     3 nodes clusters
            mno         6 nodes clusters
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting aggregating sources.")

        df.loc[df['source'].str.contains('compact|promql',case=False), ['source']]='compact'
        df.loc[df['source'].str.contains('mno',case=False), ['source']]='mno'

        self.logger.debug(f"[{inspect.stack()[0][3]}] Ending aggregating sources.")

    def get_clean_dataset(self, df):
        """
        Consolidate clusters into two groups: compact and mno
            compact     3 nodes clusters
            mno         6 nodes clusters
        """
        self.logger.debug(f"[{inspect.stack()[0][3]}] Starting cleaning dataset. Initial shape {df.shape}")

        # regular expression of substrings in columns to drop
        regex=f"yy|total_y_|run_id|_vda|_vdb|_sda|_sdb|_sr1|_sr0|_attach|_nvme|_version|_master"
        df=df[df.columns.drop(list(df.filter(regex=regex, axis=1)))]

        # drop duplicate rows
        df=df.drop_duplicates()

        self.logger.debug(f"[{inspect.stack()[0][3]}] Ending cleaning dataset. Final shape {df.shape}")
        return df

