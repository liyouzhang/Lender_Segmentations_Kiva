import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler


def convert_to_peroid(df):
    '''[summary]
    convert datatime stamp to a period until today
    Arguments:
        df {[pandas dataframe]} -- [description]
    '''
    today = df['LAST_LOGIN_DATE'].max()  # 2018-5-9
    df["last_login_today_months"] = (
        (df['LAST_LOGIN_DATE'] - today) / -np.timedelta64(1, 'M')).astype(int)
    no_nan_already_represented = [
        "VINTAGE_DATE", 'VINTAGE_YEAR', 'VINTAGE_MONTH', 'LAST_LOGIN_DATE']
    df = df.drop(no_nan_already_represented, axis=1)
    return df


def create_donation_tip_col(df):
    '''Create a new feature on donation rate. replace purchase total 0 to 0.01 to avoid infinite number in division
    '''
    df.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL = df.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL.replace({
                                                                                              0: 0.01})
    df['lifetime_ave_tip_rate'] = (
        df.LIFETIME_DONATION_TOTAL/df.LIFETIME_ACCOUNT_LOAN_PURCHASE_TOTAL)
    return df


def dummify(df, col_list=['FIRST_TIME_DEPOSITOR_REPORTING_CATEGORY',
                          'FIRST_TRANSACTION_REFERRAL',
                          'FIRST_BASKET_CATEGORY',
                          'USER_LOCATION_COUNTRY', 'FIRST_LOAN_REGION']):
    for col in col_list:
        if df[col].isnull().sum() == 0:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        else:
            dummies = pd.get_dummies(
                df[col], prefix=col, dummy_na=True, drop_first=True)
        df[dummies.columns] = dummies
    catogories_already_dummified = ['FIRST_TIME_DEPOSITOR_REPORTING_CATEGORY',
                                    'FIRST_TRANSACTION_REFERRAL',
                                    'FIRST_BASKET_CATEGORY',
                                    'USER_LOCATION_COUNTRY',
                                    'FIRST_LOAN_REGION']
    df = df.drop(catogories_already_dummified, axis=1)
    return df


def drop_columns(df):

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
                                'FIRST_DEPOSIT_DATE',
                                "ACTIVE_LIFETIME_MONTHS"]
    loan_preference = ['NUM_DISTINCT_COUNTRIES_LENT_TO',
                       'NUM_AFRICA_LOANS',
                       'NUM_ASIA_LOANS',
                       'NUM_CENTRAL_AMERICA_LOANS',
                       'NUM_EASTERN_EUROPE_LOANS',
                       'NUM_NORTH_AMERICA_LOANS',
                       'NUM_OCEANIA_LOANS',
                       'NUM_SOUTH_AMERICA_LOANS',
                       'NUM_EXPIRING_SOON_LOANS',
                       'NUM_SECTOR_AGRICULTURE_LOANS',
                       'NUM_SECTOR_TRANSPORTATION_LOANS',
                       'NUM_SECTOR_SERVICE_LOANS',
                       'NUM_SECTOR_CLOTHING_LOANS',
                       'NUM_SECTOR_HEALTH_LOANS',
                       'NUM_SECTOR_RETAIL_LOANS',
                       'NUM_SECTOR_MANUFACTURING_LOANS',
                       'NUM_SECTOR_ARTS_LOANS',
                       'NUM_SECTOR_HOUSING_LOANS',
                       'NUM_SECTOR_FOOD_LOANS',
                       'NUM_SECTOR_WHOLESALE_LOANS',
                       'NUM_SECTOR_CONSTRUCTION_LOANS',
                       'NUM_SECTOR_EDUCATION_LOANS',
                       'NUM_SECTOR_PERSONAL_USE_LOANS',
                       'NUM_SECTOR_ENTERTAINMENT_LOANS',
                       'NUM_BUNDLE_GREEN_LOANS',
                       'NUM_BUNDLE_HIGHER_ED_LOANS',
                       'NUM_BUNDLE_ISLAMIC_FINANCE_LOANS',
                       'NUM_BUNDLE_YOUTH_LOANS',
                       'NUM_BUNDLE_STARTUP_LOANS',
                       'NUM_BUNDLE_WATER_LOANS',
                       'NUM_BUNDLE_VULNERABLE_LOANS',
                       'NUM_BUNDLE_FAIR_TRADE_LOANS',
                       'NUM_BUNDLE_MOBILE_TECH_LOANS',
                       'NUM_BUNDLE_RURAL_LOANS',
                       'NUM_BUNDLE_UNDERFUNDED_LOANS',
                       'NUM_BUNDLE_CONFLICT_ZONE_LOANS',
                       'NUM_BUNDLE_JOB_CREATION_SME_LOANS',
                       'NUM_BUNDLE_GROWING_BUSINESSES_LOANS',
                       'NUM_BUNDLE_HEALTH_LOANS',
                       'NUM_BUNDLE_DISASTER_RECOVERY_LOANS',
                       'NUM_BUNDLE_INNOVATIVE_LOANS',
                       'NUM_BUNDLE_REFUGEE_LOANS',
                       'NUM_BUNDLE_SOCIAL_ENTERPRISE_LOANS',
                       'NUM_BUNDLE_CLEAN_ENERGY_LOANS',
                       'NUM_BUNDLE_SOLAR_LOANS']
    large_na_not_important = ['USER_LOCATION_STATE', 'USER_LOCATION_CITY',
                              'FIRST_LOAN_COUNTRY']
    ids = ['FUND_ACCOUNT_ID', 'LOGIN_ID']
    first_loans_nans = ['FIRST_LOAN_PURCHASE_WEIGHTED_AVERAGE_TERM',
                        'NUMBER_OF_LOANS_IN_FIRST_LOAN_CHECKOUT',
                        'NUMBER_OF_FIRST_LOANS_STILL_OUTSTANDING',
                        'PERCENT_FIRST_LOANS_EXPIRED',
                        'PERCENT_FIRST_LOANS_DEFAULTED',
                        'PERCENT_FIRST_LOANS_REPAID'
                        ]
    col_list = contain_na_but_important + \
        large_na_not_important+ids + first_loans_nans #+ loan_preference
    df = df.drop(col_list, axis=1)
    return df


def fill_cont_nans(df, num_cols=["LIFETIME_LENDER_WEIGHTED_AVERAGE_LOAN_TERM"]):
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    return df


def convert_datetime(df, col_list=[  
    #'FIRST_TRANSACTION_DATE',
    #'FIRST_DEPOSIT_DATE',
    #'LAST_TRANSACTION_DATE',
        'LAST_LOGIN_DATE']):
        for col in col_list:
            df[col] = pd.to_datetime(df[col])
        return df


# def logify(df, col_list=['ACTIVE_LIFETIME_MONTHS']):
#     for col in col_list:
#         df[col+'_log'] = np.log(df[col]+1)
#     return df

# def interactify(df, interacter1=['user_rated_driver'], interacter2=['avg_rating_of_driver']):
#     # print(type(df["user_rated_driver"]))
#     for col1, col2 in zip(interacter1, interacter2):
#         df[col1+'_'+col2] = df[col1] * df[col2]
#     return df


def convert_cat_into_int(df, col_list=['IS_CORPORATE_CAMPAIGN_USER', 'IS_FREE_TRIAL_USER']):
    '''convert categorical data into its integers (0 or 1)'''
    for col in col_list:
        df[col] = df[col].cat.codes
    return df


def feature_engineer(df):
    '''return cleaned dataframe and scaled matrix X'''
    df = drop_columns(df) #drop the columns we don't use 
    df = convert_datetime(df)
    df = convert_to_peroid(df)
    df = create_donation_tip_col(df)
    df = fill_cont_nans(df)
    df = dummify(df)
    # df = logify(df)
    df = convert_cat_into_int(df)
    scaler = StandardScaler()
    scaler.fit(df.values)
    X = scaler.transform(df.values)
    return df, X
