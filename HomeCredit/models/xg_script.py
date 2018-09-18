# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from contextlib import contextmanager
import time
import gc

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index().drop("index",axis=1)
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    df = df[df['NAME_INCOME_TYPE'] != 'Maternity leave']
    df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']
    df.loc[df.FLAG_OWN_CAR=="N","OWN_CAR_AGE"] = 0
    # Imcomplete information
#     df['incomplete'] = 1
#     df.loc[df.isnull().sum(axis=1)<35, 'incomplete'] = 0
    df['num_missing'] = df.isnull().sum(axis = 1).values
    
    # Add social class
    df["HIGH_CLASS"] = 0
    df.loc[(df["NAME_EDUCATION_TYPE"].isin(["Academic degree","Higher education"]))
        & (df["OCCUPATION_TYPE"].isin(["Accountants","Core staff","HR staff","High skill tech staff","IT staff","Managers","Medicine staff","Private service staff"])),"HIGH_CLASS"] = 1
    df["LOW_CLASS"] = 0
    df.loc[(df["NAME_EDUCATION_TYPE"].isin(["Lower secondary","Secondary / secondary special"])
        & df["OCCUPATION_TYPE"].isin(["Cleaning staff","Cooking staff","Drivers","Laborers","Low-skill Laborers","Security staff","Waiters/barmen staff"])),"LOW_CLASS"] = 1
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    # df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Estimate invalid data in feature "DAYS_EMPLOYED"
    from sklearn.linear_model import LinearRegression
    valid = df[df["DAYS_EMPLOYED"]<0]
    invalid = df[df["DAYS_EMPLOYED"]>=0]
    lr = LinearRegression()
    model = lr.fit(pd.DataFrame(valid["DAYS_BIRTH"]),valid["DAYS_EMPLOYED"])
    invalid["DAYS_EMPLOYED"] = model.predict(pd.DataFrame(invalid["DAYS_BIRTH"]))
    df["DAYS_EMPLOYED"] = pd.concat([valid,invalid]).sort_index()["DAYS_EMPLOYED"]

    # Remove outliers
    df.loc[df['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    df.loc[df['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[df['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 40, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    
    # Some simple new features (percentages)
    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INC_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1+df['CNT_CHILDREN'])
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    df['NEW_DOC_IND_AVG'] = df[docs].mean(axis=1)
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_LIVE_IND_KURT'] = df[live].kurtosis(axis=1)
    
#     df['NEW_EXT_SOURCES_MEDIAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1, skipna=True)
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1, skipna=True)
    df['NEW_EXT_SOURCES_PROD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].prod(axis=1, skipna=True, min_count=1)
    df['NEW_EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1, skipna=True)
    df['NEW_EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1, skipna=True)
    df['NEW_EXT_SOURCES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1, skipna=True)
    df['NEW_EXT_SOURCES_STD'] = df['NEW_EXT_SOURCES_STD'].fillna(df['NEW_EXT_SOURCES_STD'].mean())
#     df['NEW_EXT_SOURCES_MAD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mad(axis=1, skipna=True)
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, _ = one_hot_encoder(df, nan_as_category)
    
    cols = ['FLAG_DOCUMENT_'+str(i) for i in [2,4,5,7,9,10,12,13,14,15,17,19,20,21]]
    df = df.drop(cols,axis=1)
    
    del test_df
    gc.collect()
    return df



# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = False):
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    
    # bureau_balance.csv
    temp = pd.DataFrame(bb[['SK_ID_BUREAU', 'STATUS']].groupby('SK_ID_BUREAU').first())
    temp.columns = ["Latest_Status"]
    temp["Closed_Balance"] = bb.loc[bb['STATUS']=='C',['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').last()

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size','last']}
    bb, bb_cat1 = one_hot_encoder(bb, False)
    for col in bb_cat1:
        bb_aggregations[col] = ['sum','mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bb_agg = bb_agg.join(temp,how = 'left', on = 'SK_ID_BUREAU')
    del temp
    gc.collect()
    bb_agg['Month_closed_to_end'] = bb_agg['MONTHS_BALANCE_LAST'] - bb_agg['Closed_Balance']
    bb_agg['Non_zero_DPD_cnt'] = bb_agg[['STATUS_1_SUM', 'STATUS_2_SUM', 'STATUS_3_SUM', 'STATUS_4_SUM', 'STATUS_5_SUM']].sum(axis = 1)
    bb_agg['Non_zero_DPD_ratio'] = bb_agg[['STATUS_1_MEAN', 'STATUS_2_MEAN', 'STATUS_3_MEAN', 'STATUS_4_MEAN', 'STATUS_5_MEAN']].mean(axis = 1)
    bb_agg, bb_cat2 = one_hot_encoder(bb_agg, False)
    bb_cat = bb_cat1+bb_cat2
    
    # bureau.csv
    # Replace\remove some outliers in bureau set
    bureau.loc[bureau['AMT_ANNUITY'] > .8e8, 'AMT_ANNUITY'] = np.nan
    bureau.loc[bureau['AMT_CREDIT_SUM'] > 3e8, 'AMT_CREDIT_SUM'] = np.nan
    bureau.loc[bureau['AMT_CREDIT_SUM_DEBT'] > 1e8, 'AMT_CREDIT_SUM_DEBT'] = np.nan
    bureau.loc[bureau['AMT_CREDIT_MAX_OVERDUE'] > .8e8, 'AMT_CREDIT_MAX_OVERDUE'] = np.nan
    bureau.loc[bureau['DAYS_ENDDATE_FACT'] < -10000, 'DAYS_ENDDATE_FACT'] = np.nan
    bureau.loc[(bureau['DAYS_CREDIT_UPDATE'] > 0) | (bureau['DAYS_CREDIT_UPDATE'] < -40000), 'DAYS_CREDIT_UPDATE'] = np.nan
    bureau.loc[bureau['DAYS_CREDIT_ENDDATE'] < -10000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    bureau.drop(bureau[bureau['DAYS_ENDDATE_FACT'] < bureau['DAYS_CREDIT']].index, inplace = True)
    
    # Some new features in bureau set
    bureau['bureau1'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['bureau2'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['bureau3'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_OVERDUE']
    bureau['bureau4'] = bureau['DAYS_CREDIT'] - bureau['CREDIT_DAY_OVERDUE']
    bureau['bureau5'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['bureau6'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau7'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau8'] = bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT_ENDDATE']
    
    bureau.drop(['CREDIT_CURRENCY'], axis=1, inplace= True)
    bureau, bureau_cat = one_hot_encoder(bureau, False)
    bureau = bureau.join(bb_agg,how="left",on="SK_ID_BUREAU")
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb,bb_agg
    gc.collect()

    num_aggregations = {
        'DAYS_CREDIT': ['min','max','mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min','max','mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max','mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max','mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max','mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max','mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'MONTHS_BALANCE_LAST': ['min','max','mean'],
        'Closed_Balance': ['min','max','mean'],
        'Month_closed_to_end': ['min','max','mean'],
        'Non_zero_DPD_cnt': ['sum'],
        'Non_zero_DPD_ratio': ['mean']
    }
    for num in ['bureau'+str(i+1) for i in range(8)]:
        num_aggregations[num] = ['max','mean']
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat1: 
        cat_aggregations[cat + "_MEAN"] = ['mean']
        cat_aggregations[cat + "_SUM"] = ['sum']
    for cat in bb_cat2: 
        cat_aggregations[cat] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()  
    return bureau_agg



# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Replace some outliers
    prev.loc[prev['AMT_CREDIT'] > 6000000, 'AMT_CREDIT'] = np.nan
    prev.loc[prev['SELLERPLACE_AREA'] > 3500000, 'SELLERPLACE_AREA'] = np.nan
    prev[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
             'DAYS_LAST_DUE', 'DAYS_TERMINATION']].replace(365243, np.nan, inplace = True)
    # Some new features
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['prev1'] = prev.isnull().sum(axis = 1).values
    prev['prev2'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['prev3'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['prev4'] = prev['AMT_GOODS_PRICE'] - prev['AMT_CREDIT']
    prev['prev5'] = prev['DAYS_FIRST_DRAWING'] - prev['DAYS_FIRST_DUE']
    prev['prev6'] = (prev['DAYS_TERMINATION'] < -500).astype(int)

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    for num in ['prev'+str(i+1) for i in range(6)]:
        num_aggregations[num] = ['max','mean']
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
     # Replace some outliers
    pos.loc[pos['CNT_INSTALMENT_FUTURE'] > 60, 'CNT_INSTALMENT_FUTURE'] = np.nan
    # Some new features
    pos['pos CNT_INSTALMENT more CNT_INSTALMENT_FUTURE'] = \
                    (pos['CNT_INSTALMENT'] > pos['CNT_INSTALMENT_FUTURE']).astype(int)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'pos CNT_INSTALMENT more CNT_INSTALMENT_FUTURE': ['max','mean'],
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg



# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Replace some outliers
    ins.loc[ins['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
    ins.loc[ins['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan
    # Some new features
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['ins1'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['ins2'] = (ins['NUM_INSTALMENT_NUMBER'] == 100).astype(int)
    ins['ins3'] = (ins['DAYS_INSTALMENT'] > ins['NUM_INSTALMENT_NUMBER'] * 50 / 3 - 11500 / 3).astype(int)
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['mean','var'],
        'PAYMENT_DIFF': ['mean','var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
    }
    for num in ['ins'+str(i+1) for i in range(3)]:
        aggregations[num] = ['max','mean']
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg



# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    
    # Replace some outliers
    cc.loc[cc['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
    cc.loc[cc['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan
    # Some new features
    cc['cc1'] = cc.isnull().sum(axis = 1).values
    cc['cc2'] = cc['SK_DPD'] - cc['MONTHS_BALANCE']
    cc['cc3'] = cc['SK_DPD_DEF'] - cc['MONTHS_BALANCE']
    cc['cc4'] = cc['SK_DPD'] - cc['SK_DPD_DEF']  
    cc['cc5'] = cc['AMT_TOTAL_RECEIVABLE'] - cc['AMT_RECIVABLE']
    cc['cc6'] = cc['AMT_TOTAL_RECEIVABLE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['cc7'] = cc['AMT_RECIVABLE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['cc8'] = cc['AMT_BALANCE'] - cc['AMT_RECIVABLE']
    cc['cc9'] = cc['AMT_BALANCE'] - cc['AMT_RECEIVABLE_PRINCIPAL']
    cc['cc10'] = cc['AMT_BALANCE'] - cc['AMT_TOTAL_RECEIVABLE']
    cc['cc11'] = cc['AMT_DRAWINGS_CURRENT'] - cc['AMT_DRAWINGS_ATM_CURRENT']
    cc['cc12'] = cc['AMT_DRAWINGS_CURRENT'] - cc['AMT_DRAWINGS_OTHER_CURRENT']
    cc['cc13'] = cc['AMT_DRAWINGS_CURRENT'] - cc['AMT_DRAWINGS_POS_CURRENT']
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def write_to_csv(test_df,sub_preds,file_name="submission.csv"):
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(file_name, index= False)

def kfold_xgboost(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting XGBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    train_df = train_df.replace(np.inf, 0)
    test_df = test_df.replace(np.inf, 0)
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    dtest = xgb.DMatrix(test_df[feats],test_df["TARGET"])
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        xgb_params = {
            'seed': 0,
            'silent': 1,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'learning_rate': 0.05,
            'objective': 'binary:logistic',
            'max_depth': 8,
            'num_parallel_tree': 1,
            'min_child_weight': 30,
            'nrounds': 1000,
            'eval_metric': 'auc',
        }
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)
        del train_x, train_y
        gc.collect()
        
        watchlist = [(dtrain, 'train'),(dvalid, 'valid')]
        clf = xgb.train(xgb_params, dtrain, 1000, 
                        watchlist, verbose_eval=100, early_stopping_rounds=100)

        oof_preds[valid_idx] = clf.predict(dvalid, ntree_limit=clf.best_iteration)
        sub_pred = clf.predict(dtest, ntree_limit=clf.best_iteration)
        sub_preds += sub_pred / folds.n_splits
        auc_score = roc_auc_score(valid_y, oof_preds[valid_idx])
        
        feat_importances = []
        for ft, score in clf.get_fscore().items():
            feat_importances.append({'feature': ft, 'importance': score})
        fold_importance_df = pd.DataFrame(feat_importances)
#         fold_importance_df["feature"] = feats
#         fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, auc_score))
        del clf, valid_x, valid_y, dtrain, dvalid
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    write_to_csv(train_df,oof_preds,"oof_xgboost.csv")
    write_to_csv(test_df,sub_preds,"submission_xgboost.csv")
    # Write submission file and plot feature importance
#     if not debug:
#         test_df['TARGET'] = sub_preds
#         test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    temp = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)
    print("no. of contributing features: %d" % (len(temp[temp["importance"]>0])))
    display_importances(feature_importance_df)
    feature_importance_df.groupby("feature").mean().sort_values("importance",ascending=False)["importance"].to_csv("feature_importance.csv")

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(15, 100))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

def main(debug = False):
    num_rows = 50000 if debug else None
    with timer("Process application train and test"):
        df = application_train_test(num_rows)
        print("Application df shape:", df.shape)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Run XGBoost"):
        kfold_xgboost(df, num_folds=10, stratified=False, debug=debug)

if __name__ == "__main__":
    # submission_file_name = "submission.csv"
    with timer("Full model run"):
        main(True)