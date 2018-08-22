import pandas as pd
import numpy as np
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json


def read_full_data():
    '''[summary]
    read dataframe with column_types to reduce RAM use

    Returns:
        df[pandas dataframe]
    '''
    json1_file = open('column_types.json')
    json1_str = json1_file.read()
    column_types = json.loads(json1_str)
    # read data
    df = pd.read_csv('lender_dataset.csv',dtype=column_types)
    return df


def create_lenders_labels(df):
    '''[summary]
    create outliers labels based on different criteria

    Arguments:
        df {[pandas dataframe]} -- input data

    Returns:
        df[type] -- df with four lenders labels
    '''

    # create dollar amount outliers labels based on 0.9 quantile
    dollar_amount_outliers_list = ['LIFETIME_DONATION_TOTAL',
        'LIFETIME_DEPOSIT_TOTAL', 'LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL','FIRST_YEAR_DEPOSIT_TOTAL', 'FIRST_YEAR_LOAN_PURCHASE_TOTAL',
        'FIRST_YEAR_DONATION_TOTAL','FIRST_DAY_DEPOSIT_TOTAL', 'FIRST_DAY_LOAN_PURCHASE_TOTAL',
        'FIRST_DAY_DONATION_TOTAL']
    df['dollar_outliers?'] = np.any(df[dollar_amount_outliers_list] >= df[dollar_amount_outliers_list].quantile(0.9),axis=1)

    # create team outliers labels based on 0.99 quantile
    df['team_outliers?'] = np.any((df[['NUM_TEAM_LOANS',"NUM_TEAM_MEMBERSHIPS",'NUM_TEAM_MESSAGES']] >= df[['NUM_TEAM_LOANS',"NUM_TEAM_MEMBERSHIPS",'NUM_TEAM_MESSAGES']].quantile(0.99)),axis=1)

    # create comments outliers labels based on non_zeros
    df['comments_outliers?'] = np.any(df[['NUM_JOURNAL_COMMENTS','NUM_LOAN_COMMENTS','NUM_STATEMENT_COMMENTS']] != np.zeros((df.shape[0],3)),axis=1)

    # create user_groups who haven't donated/purchased/deposited any money
    df['dollar_zeros?'] = np.all(df[dollar_amount_outliers_list].values == np.zeros(df[dollar_amount_outliers_list].shape),axis=1)
    return df

