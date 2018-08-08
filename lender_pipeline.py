import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.style.use('ggplot')
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


def pipeline(ls):
    #replace "no" and "yes"
    ls = ls.replace({'no': 0, 'yes': 1})
    ls.IS_INACTIVE_CREDIT_DONOR = ls.IS_INACTIVE_CREDIT_DONOR.astype(int)

    #convert datetime cols
    def convert_datetime(df, col):
        df[col] = pd.to_datetime(df[col])
    ls.filter(regex=("DATE")).columns  # List of columns with "dates"
    for col in list(ls.filter(regex=("DATE")).columns):
        convert_datetime(ls, col)

    # convert timestamp --> period
    today = ls['LAST_LOGIN_DATE'].max()  # 2018-5-9
    ls["last_login_today"] = (
        (ls['LAST_LOGIN_DATE'] - today) / -np.timedelta64(1, 'D')).astype(int)
    ls['last_transaction_today'] = (
        (ls['LAST_TRANSACTION_DATE'] - today) / -np.timedelta64(1, 'D')).astype(int)
    ls['first_transaction_period'] = (
        (ls['FIRST_TRANSACTION_DATE'] - ls['VINTAGE_DATE']) / np.timedelta64(1, 'D')).astype(int)

    # ~half of the first_deposit_day is missing so we will not use it
    ls = ls.drop(['LAST_TRANSACTION_DATE', 'LAST_LOGIN_DATE',
                  'FIRST_TRANSACTION_DATE', 'VINTAGE_DATE', 'FIRST_DEPOSIT_DATE'], axis=1)

    #Create a new feature : donation rate.
    #replace purchase total 0 to 0.01 to avoid infinite number in division
    ls.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL = ls.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL.replace({
                                                                                              0: 0.01})
    ls['lifetime_ave_tip_rate'] = (
        ls.LIFETIME_DONATION_TOTAL/ls.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL)

    ls = ls.drop(['USER_LOCATION_STATE', 'USER_LOCATION_CITY',
                  'FIRST_LOAN_COUNTRY'], axis=1)

    #fill with mode for string category
    ls['FIRST_LOAN_REGION'].fillna(
        ls['FIRST_LOAN_REGION'].mode(), inplace=True)
    #fill with median for numeric categories

    def fill_cont_nans(df, col_list=['FIRST_LOAN_PURCHASE_WEIGHTED_AVERAGE_TERM',
                                     'NUMBER_OF_LOANS_IN_FIRST_LOAN_CHECKOUT',
                                     'NUMBER_OF_FIRST_LOANS_STILL_OUTSTANDING',
                                     'PERCENT_FIRST_LOANS_EXPIRED', 'PERCENT_FIRST_LOANS_DEFAULTED',
                                     'PERCENT_FIRST_LOANS_REPAID', "LIFETIME_LENDER_WEIGHTED_AVERAGE_LOAN_TERM"]):
        for col in col_list:
            df[col].fillna(df[col].median(), inplace=True)
        return df

    ls = fill_cont_nans(ls)

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
    ls = dummify(ls)
    #drop the orriginal columns
    ls = ls.drop(['FIRST_TIME_DEPOSITOR_REPORTING_CATEGORY', 'FIRST_TRANSACTION_REFERRAL',
                  'FIRST_BASKET_CATEGORY',
                  'USER_LOCATION_COUNTRY', 'FIRST_LOAN_REGION'], axis=1)

    scaler = StandardScaler()
    X = scaler.transform(ls.values)
    return X
