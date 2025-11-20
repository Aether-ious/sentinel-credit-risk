import pandas as pd
import numpy as np

class CreditRiskFeatures:
    def __init__(self):
        self.woe_mappings = {}
        self.iv_values = {}

    def calculate_woe_iv(self, df, feature, target):
        """
        Calculates WoE and IV for a categorical feature.
        """
        # Avoid division by zero
        epsilon = 1e-6
        
        # 1. Group by Feature
        lst = []
        unique_values = df[feature].unique()
        
        for val in unique_values:
            all_cnt = df[df[feature] == val].count()[feature]
            good_cnt = df[(df[feature] == val) & (df[target] == 0)].count()[feature]
            bad_cnt = df[(df[feature] == val) & (df[target] == 1)].count()[feature]
            
            lst.append({
                'Value': val,
                'Good': good_cnt,
                'Bad': bad_cnt
            })
            
        dset = pd.DataFrame(lst)
        
        # 2. Calculate Distributions
        total_good = dset['Good'].sum()
        total_bad = dset['Bad'].sum()
        
        dset['Distr_Good'] = dset['Good'] / total_good
        dset['Distr_Bad'] = dset['Bad'] / total_bad
        
        # 3. Calculate WoE
        # WoE = ln(Distr_Good / Distr_Bad)
        dset['WoE'] = np.log((dset['Distr_Good'] + epsilon) / (dset['Distr_Bad'] + epsilon))
        
        # 4. Calculate IV
        # IV = (Distr_Good - Distr_Bad) * WoE
        dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']
        
        iv_score = dset['IV'].sum()
        
        # Return the mapping dictionary
        return dset.set_index('Value')['WoE'].to_dict(), iv_score

    def fit_transform(self, df, target_col='target'):
        """
        Automatically converts all categorical columns to WoE values.
        """
        df_processed = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        print(f"⚙️  Transforming {len(categorical_cols)} categorical features using WoE...")
        
        for col in categorical_cols:
            if col == target_col: continue
            
            mapping, iv = self.calculate_woe_iv(df, col, target_col)
            
            # Save the mapping for later (inference)
            self.woe_mappings[col] = mapping
            self.iv_values[col] = iv
            
            # Apply transformation
            df_processed[col] = df_processed[col].map(mapping).fillna(0)
            
            print(f"   - {col}: IV = {iv:.4f}")
            
        return df_processed