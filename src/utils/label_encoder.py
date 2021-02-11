# +
from sklearn.preprocessing import LabelEncoder
import numpy as np

def label_encoder_target(df,target_col = 'higher_modality'):
    '''
    Function lo label encode the target
    Input
    . df         : Pandas dataframe with the target as a columns
    . target_col : Column that will be used as target
    Output:
    . le         : Label encoder scklearn function
    . dict_      : Dictionary of the label encoder 
    '''
    le = LabelEncoder()    
    unique_values  =  np.sort(df[target_col].unique())
    le.fit(unique_values)
    dict_ = {}
    for i in unique_values:
        dict_[i] = le.transform([i])[0]
    return le,dict_
# -


