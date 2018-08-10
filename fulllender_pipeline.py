import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


def convert_to_peroid(ls):
    '''[summary]
    convert datatime stamp to a period until today
    Arguments:
        ls {[pandas dataframe]} -- [description]
    '''
    today = ls['LAST_LOGIN_DATE'].max()  # 2018-5-9
    ls["last_login_today"] = (
        (ls['LAST_LOGIN_DATE'] - today) / -np.timedelta64(1, 'D')).astype(int)
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
                          'USER_LOCATION_COUNTRY','FIRST_LOAN_REGION']):
    for col in col_list:
        if df[col].isnull().sum() == 0:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        else:
            dummies = pd.get_dummies(
                df[col], prefix=col, dummy_na=True, drop_first=True)
        df[dummies.columns] = dummies
    return df


def drop_columns(ls):

    contain_na_but_important = ['LIFETIME_DEPOSIT_NUM', 
                                'LIFETIME_ACCOUNT_LOAN_PURCHASE_NUM', 
                                'LIFETIME_PROXY_LOAN_PURCHASE_NUM',
                                'LIFETIME_DONATION_NUM', 
                                'CORE_LOAN_PURCHASE_NUM', 
                                'CORE_LOAN_PURCHASE_TOTAL', 
                                'DIRECT_LOAN_PURCHASE_NUM', 
                                'DIRECT_LOAN_PURCHASE_TOTAL',
                                'LAST_TRANSACTION_DATE',
                                'FIRST_TRANSACTION_DATE',
                                'FIRST_DEPOSIT_DATE']
    catogories_already_dummified = ['FIRST_TIME_DEPOSITOR_REPORTING_CATEGORY',
                                    'FIRST_TRANSACTION_REFERRAL', 
                                    'FIRST_BASKET_CATEGORY', 
                                    'USER_LOCATION_COUNTRY', 
                                    'FIRST_LOAN_REGION']
    no_nan_already_represented = ["VINTAGE_DATE", 'VINTAGE_YEAR', 'VINTAGE_MONTH','LAST_LOGIN_DATE']
    large_na_not_important = ['USER_LOCATION_STATE', 'USER_LOCATION_CITY',
                              'FIRST_LOAN_COUNTRY']
    ids = ['FUND_ACCOUNT_ID', 'LOGIN_ID']
    col_list = contain_na_but_important+catogories_already_dummified+no_nan_already_represented+large_na_not_important+ids
    ls = ls.drop(col_list, axis=1)
    return ls


def fill_cont_nans(df, num_cols=['FIRST_LOAN_PURCHASE_WEIGHTED_AVERAGE_TERM',
                                 'NUMBER_OF_LOANS_IN_FIRST_LOAN_CHECKOUT',
                                 'NUMBER_OF_FIRST_LOANS_STILL_OUTSTANDING',
                                 'PERCENT_FIRST_LOANS_EXPIRED', 'PERCENT_FIRST_LOANS_DEFAULTED',
                                 'PERCENT_FIRST_LOANS_REPAID', "LIFETIME_LENDER_WEIGHTED_AVERAGE_LOAN_TERM"],
                   ):
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    return df


def convert_datetime(df, col_list=[#'VINTAGE_DATE',
                                   #'FIRST_TRANSACTION_DATE',
                                   #'FIRST_DEPOSIT_DATE',
                                   #'LAST_TRANSACTION_DATE',
                                   'LAST_LOGIN_DATE']):
        for col in col_list:
            df[col] = pd.to_datetime(df[col])
        return df


def logify(df, col_list=['ACTIVE_LIFETIME_MONTHS']):
    for col in col_list:
        df[col+'_log'] = np.log(df[col]+1)
    return df

# def interactify(df, interacter1=['user_rated_driver'], interacter2=['avg_rating_of_driver']):
#     # print(type(df["user_rated_driver"]))
#     for col1, col2 in zip(interacter1, interacter2):
#         df[col1+'_'+col2] = df[col1] * df[col2]
#     return df

def convert_cat_into_int(df,col_list=['IS_CORPORATE_CAMPAIGN_USER','IS_FREE_TRIAL_USER']):
    for col in col_list:
        df[col] = df[col].cat.codes
    return df


def feature_engineer(ls):
    '''return cleaned dataframe and scaled matrix X'''
    ls = convert_datetime(ls)
    ls = convert_to_peroid(ls)
    ls = create_donation_tip_col(ls)
    ls = fill_cont_nans(ls)
    ls = dummify(ls)
    ls = logify(ls)
    ls = drop_columns(ls)
    ls = convert_cat_into_int(ls)
    scaler = StandardScaler()
    scaler.fit(ls.values)
    X = scaler.transform(ls.values)
    return ls, X
