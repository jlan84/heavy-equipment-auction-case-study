import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import scipy.stats as scs
import statsmodels.api as sm

if __name__ == '__main__':
    df_train = pd.read_csv('../data/Train.zip')
    # print (df_train.head())
    # print(df_train.describe())
    # print(df_train.info())
    # print(df_train.columns)
    # print(df_train['state'].unique())
    # print(df_train['Stick'].unique())
    month_df = pd.to_datetime(df_train['saledate'])
    df_train['saledate'] = month_df
    recent_year_mask = df_train['saledate'].dt.year >=2004
    recent_sale_df = df_train[recent_year_mask]
    print(df_train.info())
    print(recent_sale_df.info())
    month_dummy_df = pd.get_dummies(month_df.dt.month)
    year_dummy_df = pd.get_dummies(month_df.dt.year)
    # print(month_dummy_df.head(10))
    # print(month_dummy_df.sum())
    recent_sale_df.to_csv('../data/recent_sales.csv')
    # print(year_dummy_df.head(10))
    # print(year_dummy_df.sum())


    month_dummy_df['saleprice'] = df_train['SalePrice']
    # print(month_dummy_df.head())
