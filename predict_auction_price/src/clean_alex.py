import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import scipy.stats as scs
import statsmodels.api as sm

def add_saleid(dummy_df, train):
    dummy_df['SalesID'] = train['SalesID'].copy()

if __name__ == '__main__':
    df_train = pd.read_csv('../data/recent_sales.csv')
    # print (df_train.head())
    # print(df_train.describe())
    # print(df_train.info())
    # print(df_train.columns)
    # print(df_train['state'].unique())
    # print(df_train['Stick'].unique())

    # print(month_dummy_df.head(10))
    # print(month_dummy_df.sum())
    # 
    # print(year_dummy_df.head(10))
    # print(year_dummy_df.sum())
    cleaning_df = df_train[['SalesID', 'SalePrice','YearMade', 'MachineHoursCurrentMeter', 'saledate', 'ProductSize', 'state', 'ProductGroupDesc', 'Drive_System']]
    cleaning_df['saledate'] = pd.to_datetime(cleaning_df['saledate'])
    # for col in cleaning_df.columns:
    #     print(str(col), ': ', cleaning_df[col].unique())
    drive_system_dummy = pd.get_dummies(cleaning_df['Drive_System']).drop('No', axis=1) #Reference = No/NaN
    add_saleid(drive_system_dummy, df_train)

    ProductGroupDesc_dummy = pd.get_dummies(cleaning_df['ProductGroupDesc']).drop('Track Excavators', axis=1) #Reference = Track Excavators
    add_saleid(ProductGroupDesc_dummy, df_train)

    state_dummy = pd.get_dummies(cleaning_df['state']).drop('Florida', axis=1) # Reference = Florida
    add_saleid(state_dummy, df_train)    
    ProductSize_dummy = pd.get_dummies(cleaning_df['ProductSize']).drop('Medium', axis=1) #Reference = Medium
    add_saleid(ProductSize_dummy, df_train)
    month_dummy_df = pd.get_dummies(cleaning_df['saledate'].dt.month).drop(3, axis=1) #Reference = Feb
    add_saleid(month_dummy_df, df_train)
    year_dummy_df = pd.get_dummies(cleaning_df['saledate'].dt.year).drop(2011, axis=1) #Reference = 2011
    add_saleid(year_dummy_df, df_train)
    print(state_dummy.head())
    full_df = df_train[['SalesID', 'SalePrice']]
    
    #day_dummy_df = pd.get_dummies(cleaning_df['saledate'].dt.dayofweek)
    # month_dummy_df['saleprice'] = df_train['SalePrice']
