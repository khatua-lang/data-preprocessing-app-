import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import json

class DataAgent:
    def __init__(self):
        self.logs = []
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []
        self.strategy = {}
        self.target_col = None
        self.id_cols = []
        self.task_type = None
        self.target_encoder = None

    def log(self, message):
        self.logs.append(message)

    def _detect_mixed_type(self, series):
        if pd.api.types.is_numeric_dtype(series):
            return False, 1.0
        numeric_coerced = pd.to_numeric(series, errors='coerce')
        valid_numeric_ratio = numeric_coerced.notna().mean()
        if valid_numeric_ratio > 0 and valid_numeric_ratio < 1.0:
            return True, valid_numeric_ratio
        return False, valid_numeric_ratio

    def profile(self, df, target_col=None, task_type='Classification'):
        self.logs.append(f"Started profiling dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        profile_data = {}
        self.target_col = target_col
        self.task_type = task_type
        self.id_cols = []
        
        for col in df.columns:
            if col == self.target_col:
                profile_data[col] = {'name': col, 'resolved_type': 'ignore', 'reason': 'Target Variable'}
                self.log(f"Column '{col}' ignored (Target Variable).")
                continue
                
            series = df[col]
            
            # Detect ID columns
            is_id_name = col.lower() in ['id', 'uuid', 'index']
            is_unique_numeric = pd.api.types.is_numeric_dtype(series) and series.nunique() == len(series) and len(series) > 0
            
            if is_id_name or is_unique_numeric:
                self.id_cols.append(col)
                profile_data[col] = {'name': col, 'resolved_type': 'ignore', 'reason': 'ID Column'}
                self.log(f"Column '{col}' ignored (ID Column detected).")
                continue

            missing_ratio = series.isna().mean()
            col_info = {'missing_ratio': missing_ratio, 'type': str(series.dtype), 'name': col}
            
            is_mixed, num_ratio = self._detect_mixed_type(series)
            col_info['is_mixed'] = is_mixed
            col_info['num_ratio'] = num_ratio
            
            if is_mixed:
                if num_ratio > 0.8:
                    col_info['resolved_type'] = 'numeric'
                    self.log(f"Column '{col}' is mixed but >80% numeric. Proposing to coerce to numeric.")
                else:
                    col_info['resolved_type'] = 'categorical'
                    self.log(f"Column '{col}' is mixed but <=80% numeric. Proposing to cast to string.")
            else:
                col_info['resolved_type'] = 'numeric' if pd.api.types.is_numeric_dtype(series) else 'categorical'
            
            if col_info['resolved_type'] == 'numeric':
                s_num = pd.to_numeric(series, errors='coerce').dropna() if is_mixed else series.dropna()
                if len(s_num) > 0:
                    q1 = s_num.quantile(0.25)
                    q3 = s_num.quantile(0.75)
                    iqr = q3 - q1
                    outlier_ratio = ((s_num < (q1 - 1.5 * iqr)) | (s_num > (q3 + 1.5 * iqr))).mean()
                    col_info['outlier_ratio'] = outlier_ratio
                    col_info['has_outliers'] = outlier_ratio > 0.05
                else:
                    col_info['outlier_ratio'] = 0
                    col_info['has_outliers'] = False
            else:
                s_cat = series.astype(str)
                unique_vals = s_cat.nunique()
                cardinality_ratio = unique_vals / len(s_cat) if len(s_cat) > 0 else 0
                col_info['unique_vals'] = unique_vals
                col_info['cardinality_ratio'] = cardinality_ratio
                col_info['is_high_cardinality'] = cardinality_ratio > 0.20
            
            profile_data[col] = col_info

        return profile_data

    def propose_strategy(self, profile_data):
        strategy = {}
        for col, info in profile_data.items():
            if info.get('resolved_type') == 'ignore':
                continue

            col_strategy = {
                'type_cast': None,
                'impute': None,
            }

            if info.get('is_mixed'):
                col_strategy['type_cast'] = 'to_numeric' if info['resolved_type'] == 'numeric' else 'to_string'

            if info.get('missing_ratio', 0) > 0:
                if info['resolved_type'] == 'numeric':
                    # Prefer median when outliers are present, mean otherwise
                    col_strategy['impute'] = 'median' if info.get('has_outliers') else 'mean'
                else:
                    col_strategy['impute'] = 'most_frequent'

            strategy[col] = col_strategy

        return strategy

    def _apply_type_casting(self, df, strategy):
        df_out = df.copy()
        for col, strat in strategy.items():
            if strat.get('type_cast') == 'to_numeric':
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
                self.log(f"Coerced column '{col}' to numeric.")
            elif strat.get('type_cast') == 'to_string':
                df_out[col] = df_out[col].astype(str).replace('nan', np.nan)
                self.log(f"Cast column '{col}' to string.")
        return df_out

    def _get_imputer(self, method):
        if method == 'mean': return SimpleImputer(strategy='mean')
        if method == 'median': return SimpleImputer(strategy='median')
        if method == 'most_frequent': return SimpleImputer(strategy='most_frequent')
        if method == 'constant': return SimpleImputer(strategy='constant', fill_value='Missing')
        if method == 'knn': return KNNImputer()
        return None



    def fit(self, df, strategy):
        self.strategy = strategy
        self.logs.append("Fitting preprocessing models...")

        df_cast = self._apply_type_casting(df, strategy)

        self.imputers = {}
        for col in df.columns:
            if col not in strategy:
                continue

            strat = strategy[col]
            imputer_method = strat.get('impute')

            if imputer_method and imputer_method != 'None':
                imputer = self._get_imputer(imputer_method)
                if imputer is not None:
                    imputer.fit(df_cast[[col]])
                    self.imputers[col] = imputer
                    self.log(f"Fitted {imputer_method} imputer on '{col}'.")

        self.logs.append("Fitting complete.")
        return self

    def transform(self, df):
        self.logs.append("Transforming data...")
        df_out = self._apply_type_casting(df, self.strategy)

        for col in df_out.columns:
            # Pass target column through untouched
            if getattr(self, 'target_col', None) == col:
                continue

            # Skip columns not in strategy (e.g. ID columns)
            if col not in self.strategy:
                continue

            # Apply imputation if a fitted imputer exists for this column
            if col in self.imputers:
                imputed = self.imputers[col].transform(df_out[[col]])
                df_out[col] = imputed[:, 0]
                self.log(f"Imputed missing values in '{col}'.")

        self.logs.append("Transformation complete.")
        return df_out

    def save_pipeline(self, filepath_pkl):
        """Persist the fitted preprocessor to disk."""
        joblib.dump({
            'strategy': self.strategy,
            'imputers': self.imputers,
            'target_col': self.target_col,
            'id_cols': self.id_cols,
        }, filepath_pkl)
        self.log(f"Saved preprocessor pipeline to {filepath_pkl}")

    def get_logs(self):
        return self.logs
