import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_DIR = "data/"
NAMES = {
    "CpuName" : "cpu",
    "CpuNumberOfCores" : "cpu_cores",
    "CpuNumberOfThreads" : "cpu_threads",
    "CpuBaseClock" : "cpu_clock",
    "CpuCacheL1" : "cpu_L1",
    "CpuCacheL2" : "cpu_L2",
    "CpuCacheL3" : "cpu_L3",
    "CpuDieSize" : "cpu_die",
    "CpuFrequency" : "cpu_frq",
    "CpuMultiplier" : "cpu_multiplier",
    "CpuMultiplierUnlocked" : "cpu_has_oc",
    "CpuProcessSize" : "cpu_prcss",
    "CpuTDP" : "cpu_tdp",
    "CpuNumberOfTransistors" : "cpu_trnsstrs",
    "CpuTurboClock" : "cpu_tclock"
}

class Preprocess():

    def num_to_nom_cat_bin(self, df, feature, bin):
        return df[feature].apply(lambda val: bin[0] if val == 0 else bin[1])

    def change_datatype(self, df, feature, type):
        return df[feature].astype(type)
    
    def divide_by(self, df, feature, modifier):
        return df[feature].apply(lambda val: np.round(val/modifier, 6))
    
    def remove_duplicates(self, df):
        df_cp = df.copy()
        return df_cp.drop_duplicates()
    
    def load_data(self, csv_file_name):
        csv = pd.read_csv(DATA_DIR+csv_file_name, na_values=["?"])

        cpu_columns = csv.columns.to_list()
        cpu_columns = cpu_columns[1:16]

        df = csv[cpu_columns].copy()
        return df
    
    def value_replace(self, df, column, to_replace, value):
        return df[column].replace(to_replace=to_replace, value=value)

    def propagate_valid(self, df, ff_col, order_col, ascending):
        sorted_df = df.sort_values(by=order_col, ascending=ascending)
        return sorted_df[ff_col].ffill()

    def get_feature_outliers(self, df, feature, multiplier=1.5):
        perc25 = df[feature].quantile(0.25)
        perc75 = df[feature].quantile(0.75)
        iqr = perc75-perc25
        above= perc75 + multiplier * iqr
        below = perc25 - multiplier * iqr
        outlier_rows = df[(df[feature]>above) | (df[feature]<below)]
        return outlier_rows.shape[0]

    def remove_features(self, df, columns):
        df = df.drop(columns=columns, axis=1)
        return df

    def rename_features(self, df):
        df = df.rename(columns=NAMES)
        return df

    def split_data(self, df):
        train, test_vldtn = train_test_split(df, test_size=0.4, random_state=42)
        vldtn, test = train_test_split(test_vldtn, test_size=0.5, random_state=42)
        return train, vldtn, test

    def restart_df_index(self, df):
        return df.reset_index(drop=True)

    def isolate_target_feature(self, df, feature):
        return df[feature].values

    def remove_from_split(self, df, feature):
        del df[feature]