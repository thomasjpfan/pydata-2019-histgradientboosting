from urllib.request import urlretrieve
import os
from gzip import GzipFile
from time import time
import argparse

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.utils import (
    get_equivalent_estimator)


HERE = os.path.dirname(__file__)
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
       "HIGGS.csv.gz")
m = Memory(location='/tmp', mmap_mode='r')


@m.cache
def load_data():
    filename = os.path.join(HERE, URL.rsplit('/', 1)[-1])
    if not os.path.exists(filename):
        print(f"Downloading {URL} to {filename} (2.6 GB)...")
        urlretrieve(URL, filename)
        print("done.")

    print(f"Parsing {filename}...")
    tic = time()
    with GzipFile(filename) as f:
        df = pd.read_csv(f, header=None, dtype=np.float32)
    toc = time()
    print(f"Loaded {df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('library', choices=['sklearn', 'lightgbm',
                                            'xgboost', 'catboost'])
    parser.add_argument('--n-trees', type=int, default=100)

    args = parser.parse_args()

    n_trees = args.n_trees

    df = load_data()
    target = df.values[:, 0]
    data = np.ascontiguousarray(df.values[:, 1:])
    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=.2, random_state=0)

    n_samples, n_features = data_train.shape
    print(f"Training set with {n_samples} records with {n_features} features.")

    est = HistGradientBoostingClassifier(loss='binary_crossentropy',
                                         max_iter=n_trees,
                                         n_iter_no_change=None,
                                         random_state=0,
                                         verbose=1)

    if args.library == 'sklearn':
        print("Fitting a sklearn model...")
        tic = time()
        est.fit(data_train, target_train)
        toc = time()
        predicted_test = est.predict(data_test)
        predicted_proba_test = est.predict_proba(data_test)
        roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
        acc = accuracy_score(target_test, predicted_test)
        print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, "
              f"ACC: {acc :.4f}")

    elif args.library == 'lightgbm':
        print("Fitting a LightGBM model...")
        tic = time()
        lightgbm_est = get_equivalent_estimator(est, lib='lightgbm')
        lightgbm_est.fit(data_train, target_train)
        toc = time()
        predicted_test = lightgbm_est.predict(data_test)
        predicted_proba_test = lightgbm_est.predict_proba(data_test)
        roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
        acc = accuracy_score(target_test, predicted_test)
        print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, "
              f"ACC: {acc :.4f}")

    elif args.library == 'xgboost':
        print("Fitting an XGBoost model...")
        tic = time()
        xgboost_est = get_equivalent_estimator(est, lib='xgboost')
        xgboost_est.fit(data_train, target_train)
        toc = time()
        predicted_test = xgboost_est.predict(data_test)
        predicted_proba_test = xgboost_est.predict_proba(data_test)
        roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
        acc = accuracy_score(target_test, predicted_test)
        print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, "
              f"ACC: {acc :.4f}")

    else:  # catboost
        print("Fitting a Catboost model...")
        tic = time()
        catboost_est = get_equivalent_estimator(est, lib='catboost')
        catboost_est.fit(data_train, target_train)
        toc = time()
        predicted_test = catboost_est.predict(data_test)
        predicted_proba_test = catboost_est.predict_proba(data_test)
        roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])
        acc = accuracy_score(target_test, predicted_test)
        print(f"done in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, "
              f"ACC: {acc :.4f}")
