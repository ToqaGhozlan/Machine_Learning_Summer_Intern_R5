import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DegreeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, coord_cols, bearing_col='bearing'):
        self.coord_cols = coord_cols
        self.bearing_col = bearing_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.coord_cols:
            X[col] = np.degrees(X[col])
        if self.bearing_col in X.columns:
            X[self.bearing_col] = np.degrees(X[self.bearing_col])
            X[self.bearing_col] = (X[self.bearing_col] + 360) % 360
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, period_map):
        # e.g. {'hour': 24, 'weekday': 7, 'month': 12, 'bearing': 360}
        self.period_map = period_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, period in self.period_map.items():
            X[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / period)
            X[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / period)
        X = X.drop(columns=list(self.period_map.keys()))
        return X


class PercentileCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lower_pct=0.01, upper_pct=0.99):
        self.columns = columns
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        self.bounds_ = {}
        for col in self.columns:
            lower = X[col].quantile(self.lower_pct)
            upper = X[col].quantile(self.upper_pct)
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            X[col] = X[col].clip(lower, upper)
        return X
