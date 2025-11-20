import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass

    def calculate_woe_iv(self, df, feature, target):
        """
        Calculates Weight of Evidence (WoE) and Information Value (IV).
        Crucial for Credit Risk interpretability.
        """
        lst = []
        for i in range(df[feature].nunique()):
            val = list(df[feature].unique())[i]
            lst.append({
                'Value': val,
                'All': df[df[feature] == val].count()[feature],
                'Good': df[(df[feature] == val) & (df[target] == 0)].count()[feature],
                'Bad': df[(df[feature] == val) & (df[target] == 1)].count()[feature]
            })
        
        dset = pd.DataFrame(lst)
        dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
        dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
        dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])
        dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
        
        return dset

    def fit_transform(self, df):
        # Example: Create Debt-to-Income Ratio
        df['dti_ratio'] = df['total_debt'] / df['total_income']
        return df