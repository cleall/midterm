import pickle

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor

def create_pipeline():
    pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(
            max_depth=10,
            n_estimators=100,
            min_samples_split=2,
            min_samples_leaf=2,
            random_state=42
        )
    )
    return pipeline

def save_pipeline(pipeline, filename):
    with open(filename, "wb") as outfile:
        pickle.dump(pipeline, outfile)
    print(f"Saved pipeline check cwd for: {filename}")