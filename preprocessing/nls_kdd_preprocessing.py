import pandas as pd
import numpy as np
import os
import random
# from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from config import *


def prep_nls_kdd_train_val(val_split=0.2, cols=None, cat_cols=None,
                  ohe=None, exclude_cat=False, preprocessing=None):
    df_all = pd.read_csv(nls_kdd_datapath + "KDDTrain+.txt", header=None, names=nls_kdd_cols)
    cont_cols = [x for x in cols if x not in nls_kdd_cat_cols]

    if preprocessing == 'log':
        for c in cont_cols:
            if (c != 'label') & (c != 'difficulty'):
                df_all[c] = df_all[c].apply(lambda x: np.log(x + 1))

    if exclude_cat:
        df_encoded = df_all[cont_cols]
        ohe = False
    else:
        sorted_cols = cont_cols + cat_cols
        df_all = df_all[sorted_cols]

        df_cat = df_all[cat_cols].copy()
        ohe = OneHotEncoder()
        array_hot_encoded = ohe.fit_transform(df_cat)
        df_hot_encoded = pd.DataFrame(array_hot_encoded.toarray(), index=df_cat.index)

        df_other_cols = df_all.drop(columns=cat_cols, axis=1)
        df_encoded = pd.concat([df_other_cols, df_hot_encoded], axis=1)

    df_encoded['label'] = df_encoded['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df_encoded = df_encoded[df_encoded['label'] == 0]
    df_encoded = df_encoded.drop('difficulty', axis=1)

    trainx, valx, trainy, valy = train_test_split(df_encoded.drop(columns='label'), df_encoded['label'],
                                                  test_size=val_split)

    return trainx.values, valx.values, ohe

def prep_nls_kdd_test(cols=None, cat_cols=None,
             ohe=None, exclude_cat=False, preprocessing=None):
    df_all = pd.read_csv(nls_kdd_datapath + "/KDDTest+.txt", header=None, names=cols)
    cont_cols = [x for x in cols if x not in nls_kdd_cat_cols]
    if preprocessing == 'log':
        for c in cont_cols:
            if (c != 'label') & (c != 'difficulty'):
                df_all[c] = df_all[c].apply(lambda x: np.log(x + 1))

    if exclude_cat:
        df_encoded = df_all[cont_cols]

    else:
        sorted_cols = cont_cols + cat_cols
        df_all = df_all[sorted_cols]

        df_cat = df_all[cat_cols].copy()
        array_hot_encoded = ohe.transform(df_cat)
        df_hot_encoded = pd.DataFrame(array_hot_encoded.toarray(), index=df_cat.index)

        df_other_cols = df_all.drop(columns=cat_cols, axis=1)
        df_encoded = pd.concat([df_other_cols, df_hot_encoded], axis=1)

    original_labels = df_encoded['label']
    df_encoded['label'] = df_encoded['label'].apply(lambda x: 0 if x == 'normal' else 1)
    labels = df_encoded['label'].values
    df_encoded = df_encoded.drop(['label', 'difficulty'], axis=1)

    return df_encoded.values, labels, original_labels
