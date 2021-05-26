import seaborn as sns
import os
from pydataset import data
from scipy import stats
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# ignore warnings
import warnings
warnings.filterwarnings("ignore")



def clean_data(df):
    '''
    This function will drop payment_type_id', 'internet_service_type_id','contract_type_id', 
    convert all the columns that have yes/no to 0/1, 
    create dummy vars from 'gender', 'contract_type', 'internet_service_type', 'payment_type',
    change total_charges to a float type. 
    (remove spaces and -), lower case for all column names
    '''

    #clean data
    # conver total_charges to float
    df['total_charges'][df['total_charges']== ' ']= df['total_charges'][df['total_charges']== ' '].replace(' ', '0')
    df['total_charges'] = df['total_charges'].astype('float')
    
    #convert all the columns that have yes/no to 0/1
    col_list = ['partner', 'dependents','phone_service', 'paperless_billing','churn' ]
    df[col_list] = (df[col_list] == 'Yes').astype(int)
    
    #change columns to 0,1,2
    #getting a list of the  columns that I want to change
    col_list = list(df.select_dtypes('object').columns)[1:]
    #create a dicttionary to change the value
    var= {
        'No':0,
        'Yes':1,
        'No internet service':3
    }
    #use a for loop to change every column
    for col in col_list[2:8]:
      df[col]= df[col].map(var) 
    
    #replace the values of multiple_lines
    df.replace({'multiple_lines': {'No':1, 'Yes':2, 'No phone service': 0}}, inplace=True)
    
    #create a dummy df
    col_list = list(df.select_dtypes('object').columns)[1:]
    #create a dummy df
    for col in col_list:
        dummy_df = pd.get_dummies(df[col])
         ## Concatenate the dummy_df dataframe above with the original df
        df = pd.concat([df, dummy_df], axis=1)
    # drop the columns that we already use to create dummy_df
    df.drop(columns=col_list, inplace=True)
    
    #drop duplicates columns
    df.drop(columns = ['payment_type_id', 'internet_service_type_id','contract_type_id'], inplace=True)
    
    #  rename the column as has_internet
    df.rename(columns={'None':'has_internet'}, inplace= True )
    #changing the values to undestand better the meaning
    df['has_internet'] = df['has_internet'].replace({0: 1, 1: 0})
    # columns name change (remove space and -)
    df_clean.columns = [col.lower().replace(' ', '_').replace('-','_') for col in df_clean]
    df_clean.columns
    return df
