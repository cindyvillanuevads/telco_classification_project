import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ignore warnings
import warnings
warnings.filterwarnings("ignore")



def clean_data(df):
    '''
    This function will drop payment_type_id', 'internet_service_type_id','contract_type_id', 
    convert all the columns that have yes/no to 0/1, 
    create dummy vars from 'gender', 'contract_type', 'internet_service_type', 'payment_type',
    change total_charges to a float type. 
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
    
    # rename 'tenure' so it can be more clear that the tenure is in months
    df.rename(columns={'tenure':'tenure_months'}, inplace= True)
    #  rename the column as has_internet
    df.rename(columns={'None':'has_internet'}, inplace= True )
    #changing the values to undestand better the meaning
    df['has_internet'] = df['has_internet'].replace({0: 1, 1: 0})
    # columns name change (remove space and -)
    df.columns = [col.lower().replace(' ', '_').replace('-','_') for col in df]
    df.columns
    return df


def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on churn.
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn)
    return train, validate, test

def prepare (df):
    '''
    take in a DataFrame and use two functions
    1. clean data:
    drop duplicates, change total_charges to a float type, rename columns, convert all the columns that have yes/no to 0/1, 
    drop payment_type_id', 'internet_service_type_id','contract_type_id', create dummy vars from 'gender', 'contract_type',
    'internet_service_type', 'payment_type'
    2. split
    return train, validate, and test DataFrames; stratify on churn
    Example:
    train, validate, test = .prepare(df)
    '''
    clean_df =clean_data(df)
    train, validate, test = split_data(clean_df)

    return train, validate, test



# plot distributions
def distribution (df):
    cols =df.columns.to_list()
    for col in cols:
        if df[col].dtype != 'object':
            plt.hist(df[col])
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.ylabel('Counts of customers')
            plt.show()


# plot distributions
def distribution_obj (df):
    cols =df.columns.to_list()
    for col in cols[1:]:
        if df[col].dtype == 'object':
            plt.hist(df[col])
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.ylabel('Counts of customers')
            plt.show()