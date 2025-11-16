import numpy as np

from preprocess.preprocess import Preprocess
from pipeline.pipeline_do import (
    create_pipeline, save_pipeline
)

def main():
    pre_proc = Preprocess()
    #Load data from csv
    csv_name = "fps_videogames.csv"
    cpu_df = pre_proc.load_data(csv_name)
    #rename selected features
    rn_df = pre_proc.rename_features(cpu_df)
    #remove cpu feature
    rf_df = pre_proc.remove_features(rn_df, ["cpu"])
    #replace missing values
    rf_df["cpu_L3"] = pre_proc.value_replace(rf_df, "cpu_L3", np.nan, value=0)
    rf_df["cpu_die"] = pre_proc.propagate_valid(rf_df, "cpu_die", "cpu_prcss", True)
    rf_df["cpu_trnsstrs"] = pre_proc.propagate_valid(rf_df, "cpu_trnsstrs", "cpu_prcss", True)
    #change datatypes
    rf_df["cpu_L3"] = pre_proc.change_datatype(rf_df, "cpu_L3", int)
    rf_df["cpu_trnsstrs"] = pre_proc.change_datatype(rf_df, "cpu_trnsstrs", "int")
    #remove duplicates
    rd_df = pre_proc.remove_duplicates(rf_df)
    print(f"Dataset available observations after duplicate removal: {rd_df.shape[0]}")
    #turn numeric feature to nominal categorical
    rd_df["cpu_has_oc"] = pre_proc.num_to_nom_cat_bin(rd_df, "cpu_has_oc", ["Y", "N"])
    rd_df["cpu_has_oc"] = pre_proc.change_datatype(rd_df, "cpu_has_oc", "object")
    #change feature units
    to_ghz = ["cpu_clock", "cpu_frq", "cpu_tclock"]
    for feature in to_ghz:
        rd_df[feature] = pre_proc.divide_by(rd_df, feature, 1_000)
    
    to_mb = ["cpu_L1", "cpu_L2", "cpu_L3"]
    for feature in to_mb:
        rd_df[feature] = pre_proc.divide_by(rd_df, feature, 1_000)
    
    rd_df["cpu_die"] = pre_proc.divide_by(rd_df, "cpu_die", 1_000_000)
    rd_df["cpu_trnsstrs"] = pre_proc.divide_by(rd_df, "cpu_trnsstrs", 1_000)
    #outliers in numeric feature
    num_features = list(rd_df.select_dtypes(include=["int64", "float64"]).columns)
    mult = 3 #iqr default is 1.5 increased to 3 to allow robustness at some level
    tot_out = 0
    for ftr in num_features:
        feature_outliers = pre_proc.get_feature_outliers(rd_df, ftr, multiplier=mult)
        print(f"Outliers in {ftr}: {feature_outliers}")
        tot_out += feature_outliers
    print(f"Outliers from all features: {tot_out} with multiplier: {mult}")
    #split data
    train, vldtn, test = pre_proc.split_data(rd_df)
    #reset indexes
    train = pre_proc.restart_df_index(train)
    vldtn = pre_proc.restart_df_index(vldtn)
    test = pre_proc.restart_df_index(test)
    #isolate target feature
    target_feature = "cpu_tclock"
    y_train = pre_proc.isolate_target_feature(train, target_feature)
    y_vldtn = pre_proc.isolate_target_feature(vldtn, target_feature)
    y_test = pre_proc.isolate_target_feature(test, target_feature)
    #remove target feature from splits
    pre_proc.remove_from_split(train, target_feature)
    pre_proc.remove_from_split(vldtn, target_feature)
    pre_proc.remove_from_split(test, target_feature)
    #create pipeline
    cpu_tf_pl = create_pipeline()
    #fit pipeline to train data
    selected_features = ["cpu_cores", "cpu_threads", "cpu_frq", "cpu_multiplier",
        "cpu_tdp", "cpu_prcss", "cpu_die", "cpu_has_oc"]
    x_train_dct = train[selected_features].to_dict(orient='records')
    cpu_tf_pl.fit(x_train_dct, y_train)
    #save pipeline
    save_pipeline(cpu_tf_pl, "cpu_tf_rfpl_v1.bin")

if __name__ == "__main__":
    main()