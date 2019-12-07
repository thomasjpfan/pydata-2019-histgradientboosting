# Deep Dive into scikit-learn's HistGradientBoosting Classifier and Regressor

- [Link to slides](https://github.com/thomasjpfan/pydata-2019-histgradientboosting/blob/master/presentation.pdf)
- [Link to youtube](https://youtu.be/J9QQ6l_HToU)

## Running benchmarks

0. Install anaconda

1. Setup environment

```bash
conda env create -f environment.yml
conda activate 2019-pydata-nyc-hist
```

2. Run benchmarks for each library

First run will download the HIGGS dataset which is 2.6 GB!

```bash
# This is the number of cores (no hyperthreading)
export OMP_NUM_THREADS=12
python bench.py sklearn
python bench.py catboost
python bench.py lightgbm
python bench.py xgboost
```
