import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

######################################################################################################
#            Class to store and apply data cleaning transformations to any test set                  #
######################################################################################################
class data_cleaner:
    def __init__(self):
        self.labels               = []
        self.target               = []
        
        self.feature_list         = []
        self.bands                = {}
        self.median               = {}
        self.most_freq_class      = {}
        self.replacements         = {}
        
        self.filter_upper_values  = {} 
        self.filter_lower_values  = {}
        self.clip_lower_values    = {}
        self.clip_upper_values    = {}

    # Subset the input dataset with list of features owned by the cleaner  
    def select_features(self, df, target=None):
        if target not in df.columns:
            feature_list = set(self.feature_list) - set(['target'])
        else:
            feature_list = set(self.feature_list)
            
        return (df[list(feature_list)])
   
    # Apply the same filter and clip transformations as in the training set 
    def filter_and_clip_values(self, df):
        
        # Filter values that are not feasible 
        for key in self.filter_lower_values:  
            if key in df.columns.tolist():
                df[key][ df[key] < self.filter_lower_values[key] ] = np.nan
        for key in self.filter_upper_values: 
            if key in df.columns.tolist():
                df[key][ df[key] > self.filter_upper_values[key] ] = np.nan
            
        # Clip values that, although feasible, lead to long tails 
        for key in self.clip_lower_values:
            if key in df.columns.tolist():
                df[key].clip(lower=self.clip_lower_values[key], inplace=True)
        for key in self.clip_upper_values: 
            if key in df.columns.tolist():
                df[key].clip(upper=self.clip_upper_values[key], inplace=True)
            
        return(df)
    
    # Band numerical variables contained in bands, thus become categorical
    def band_features(self, df): 
        for key in self.bands:
            df[key] = pd.cut(df[key], self.bands[key])   
            df[key] = df[key].astype("object")
                        
        return df    
    
    # Impute median values for missings in the numerical variables
    def impute_numerical_variables(self, df):
        for key in self.median:
            if df[key].isnull().any():
                df[key].replace(np.NaN, self.median[key], inplace=True)
                assert(not df[key].isna().any())
        return df
    
    
    # Impute categorical values for missings in the numerical variables
    def impute_categorical_variables(self, df):
        for key in self.most_freq_class:
            if df[key].isnull().any():
                df[key].replace(np.NaN, self.most_freq_class[key], inplace=True)  
                assert(not df[key].isna().any())
        return df
                        
######################################################################################################
#              Class to store and apply model pre-processing transformations                         #
######################################################################################################

class data_preprocessor:
    def __init__(self):
        self.feature_encoded_list = []
        self.columns_to_scale     = []
        self.scaler               = MinMaxScaler()
        
        
    def apply_scaler(self, df):
        # Apply min_max scaler and recover a pandas DataFrame
        df[self.columns_to_scale] = self.scaler.transform(df[self.columns_to_scale])

        return df 
        
    
    def one_hot_encode(self, df):
        # One-hot encode the data and recover original columns order
        numerical_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols  = df.select_dtypes(exclude=[np.number]).columns.tolist()

        cat_enc_data = df[categorical_cols]
        # One-hot encode categorical data
        for col in categorical_cols:
            cat_enc_data = pd.concat([cat_enc_data, pd.get_dummies(cat_enc_data[col], prefix=col)],axis=1)
            cat_enc_data = cat_enc_data.drop(col, axis=1)
    
        cat_enc_data.columns = [x.replace('<','.lt.').replace('>','.gt.').replace('[','(').replace(']',')')  for x in cat_enc_data.columns]
        encoded_data = pd.concat([df[numerical_cols], cat_enc_data],axis=1)
        
        # Add encoded features that do not appear in train/test but are present in test/train, respectively. 
        diff1 = set(self.feature_encoded_list) - set(encoded_data.columns) 
        print("diff1 = ", diff1)
        for var in diff1:
            encoded_data[var] = 0
 
        diff2= set(encoded_data.columns) - set(self.feature_encoded_list) 
        print("diff2 = ", diff2)
        for var in diff2:
            encoded_data.drop(var, axis=1, inplace=True)
    
        # Unify column identifiers
        encoded_data = encoded_data[self.feature_encoded_list]
                
        return encoded_data