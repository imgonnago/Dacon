#train.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from data import build_training_data


def fit(df_train,model):

    X = df_train[["b_t", "b_t_1", "a_t_lag", "max_corr", "best_lag"]]

    y = df_train["target"]

    model.fit(X, y)
    return model

def predict(pivot, pairs, reg):
    months = pivot.columns.to_list()
    n_months = len(months)

    t_last = n_months - 1
    t_prev = n_months - 2

    preds = []

    for row in tqdm(pairs.itertuples(index=False)):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        X_test = np.array([[b_t, b_t_1, a_t_lag, corr, float(lag)]])
        y_pred = reg.predict(X_test)[0]

        # value값 log1p 에서 역변환
        y_pred = np.expm1(y_pred)


        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))



        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred


def predict_ensemble(pivot, pairs,
                     model_xgb, model_extra, model_cat,
                     w_xgb = 0.3, w_extra = 0.2, w_cat = 0.5):

    months = pivot.columns.to_list()
    n_months = len(months)
    t_last = n_months - 1
    t_prev = n_months - 2
    preds = []

    for row in tqdm(pairs.itertuples(index=False), desc="Ensemble Predict"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot.index or follower not in pivot.index:
            continue


        if t_last - lag < 0:
            continue

        idx = t_last + 1 - lag
        if idx < 0: continue

        leader_val_now = pivot.loc[leader].values[idx]

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]


        X_test = pd.DataFrame([{
            "b_t": b_t,
            "b_t_1": b_t_1,
            "a_t_lag": a_t_lag,
            "max_corr": corr,
            "best_lag": float(lag)
        }])

        pred_xgb_log = model_xgb.predict(X_test)[0]
        pred_extra_log = model_extra.predict(X_test)[0]
        pred_cat_log = model_cat.predict(X_test)[0]

        val_xgb = np.expm1(pred_xgb_log)
        val_extra = np.expm1(pred_extra_log)
        val_cat = np.expm1(pred_cat_log)

        final_val = (val_xgb * w_xgb) + (val_extra * w_extra) + (val_cat * w_cat)

        final_val = max(0.0, float(final_val))
        final_val = int(round(final_val))

        if leader_val_now == 0:
            final_val = 0

        elif final_val < 15:
            final_val = 0

        else:
            final_val = int(round(final_val))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": final_val,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred