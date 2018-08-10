import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


def convert_dtype(ls):
    ls = ls.replace({'no': 0, 'yes': 1})
    ls.IS_INACTIVE_CREDIT_DONOR = ls.IS_INACTIVE_CREDIT_DONOR.astype(int)
    return ls


def convert_to_peroid(ls):
    '''[summary]
    convert datatime stamp to a period until today
    Arguments:
        ls {[pandas dataframe]} -- [description]
    '''
    today = ls['LAST_LOGIN_DATE'].max()  # 2018-5-9
    ls["last_login_today"] = (
        (ls['LAST_LOGIN_DATE'] - today) / -np.timedelta64(1, 'D')).astype(int)
    ls['last_transaction_today'] = (
        (ls['LAST_TRANSACTION_DATE'] - today) / -np.timedelta64(1, 'D')).astype(int)
    ls['first_transaction_period'] = (
        (ls['FIRST_TRANSACTION_DATE'] - ls['VINTAGE_DATE']) / np.timedelta64(1, 'D')).astype(int)
    return ls


def create_donation_tip_col(ls):
    '''Create a new feature on donation rate. replace purchase total 0 to 0.01 to avoid infinite number in division
    '''
    ls.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL = ls.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL.replace({
                                                                                              0: 0.01})
    ls['lifetime_ave_tip_rate'] = (
        ls.LIFETIME_DONATION_TOTAL/ls.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL)
    return ls


def dummify(df, col_list=['FIRST_TIME_DEPOSITOR_REPORTING_CATEGORY',
                          'FIRST_TRANSACTION_REFERRAL',
                          'FIRST_BASKET_CATEGORY',
                          'USER_LOCATION_COUNTRY']):
    for col in col_list:
        if df[col].isnull().sum() == 0:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        else:
            dummies = pd.get_dummies(
                df[col], prefix=col, dummy_na=True, drop_first=True)
        df[dummies.columns] = dummies
    return df


def drop_columns(ls, col_list=['FIRST_TIME_DEPOSITOR_REPORTING_CATEGORY', 'FIRST_TRANSACTION_REFERRAL', 'FIRST_BASKET_CATEGORY', 'USER_LOCATION_COUNTRY', 'FIRST_LOAN_REGION', 'USER_LOCATION_STATE', 'USER_LOCATION_CITY',
                               'FIRST_LOAN_COUNTRY', 'LAST_TRANSACTION_DATE', 'LAST_LOGIN_DATE',
                               'FIRST_TRANSACTION_DATE', 'VINTAGE_DATE', 'FIRST_DEPOSIT_DATE', 'FUND_ACCOUNT_ID', 'LOGIN_ID','VINTAGE_YEAR','VINTAGE_MONTH',"ACTIVE_LIFETIME_MONTHS"]):
    ls = ls.drop(col_list, axis=1)
    return ls


def fill_cont_nans(df, num_cols=['FIRST_LOAN_PURCHASE_WEIGHTED_AVERAGE_TERM',
                                 'NUMBER_OF_LOANS_IN_FIRST_LOAN_CHECKOUT',
                                 'NUMBER_OF_FIRST_LOANS_STILL_OUTSTANDING',
                                 'PERCENT_FIRST_LOANS_EXPIRED', 'PERCENT_FIRST_LOANS_DEFAULTED',
                                 'PERCENT_FIRST_LOANS_REPAID', "LIFETIME_LENDER_WEIGHTED_AVERAGE_LOAN_TERM"],
                   str_cols=['FIRST_LOAN_REGION']):
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in str_cols:
        df[col].fillna(df[col].mode(), inplace=True)
    return df


def convert_datetime(df, col_list=['VINTAGE_DATE',
                                   'FIRST_TRANSACTION_DATE',
                                   'FIRST_DEPOSIT_DATE',
                                   'LAST_TRANSACTION_DATE',
                                   'LAST_LOGIN_DATE']):
        for col in col_list:
            df[col] = pd.to_datetime(df[col])
        return df


def logify(df, col_list=['ACTIVE_LIFETIME_MONTHS']):
    for col in col_list:
        df[col+'_log'] = np.log(df[col]+1)
    return df

def interactify(df, interacter1=['user_rated_driver'], interacter2=['avg_rating_of_driver']):
    # print(type(df["user_rated_driver"]))
    for col1, col2 in zip(interacter1, interacter2):
        df[col1+'_'+col2] = df[col1] * df[col2]
    return df

def feature_engineer(ls):
    '''return cleaned dataframe and scaled matrix X'''
    ls = convert_dtype(ls)
    ls = convert_datetime(ls)
    ls = convert_to_peroid(ls)
    ls = create_donation_tip_col(ls)
    ls = fill_cont_nans(ls)
    ls = dummify(ls)
    ls = logify(ls)
    ls = drop_columns(ls)
    scaler = StandardScaler()
    scaler.fit(ls.values)
    X = scaler.transform(ls.values)
    return ls, X
