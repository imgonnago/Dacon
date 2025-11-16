#train.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import data_load, data_preparing, find_comovement_pairs, build_training_data

def create_train():
    train = data_load()
    monthly, pivot_df_value, pivot_df_weight = data_preparing(train)
    pairs = find_comovement_pairs(pivot_df_value)

    df_train_model = build_training_data(pivot_df_value, pivot_df_weight,pairs)
    print('생성된 학습 데이터의 shape :', df_train_model.shape)
    df_train_model.head()

    return df_train_model


def fit(model,df_train):
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag','a_t_lag_weight', 'max_corr', 'best_lag']

    train_X = df_train[feature_cols]
    train_y = df_train["target"]

    model.fit(train_X, train_y)

    return model


def predict(pivot_df_value, pivot_df_weight, pairs, reg, x_scaler, y_scaler):

    cols_to_scale = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight']

    months = pivot_df_value.columns.to_list()
    n_months = len(months)
    t_last = n_months - 1
    t_prev = n_months - 2

    preds = []

    for row in tqdm(pairs.itertuples(index=False)):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot_df_value.index or follower not in pivot_df_value.index:
            continue

        a_series = pivot_df_value.loc[leader].values.astype(float)
        b_series = pivot_df_value.loc[follower].values.astype(float)
        a_weight_series = pivot_df_weight.loc[leader].values.astype(float)

        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]
        a_t_lag_weight = a_weight_series[t_last - lag]

        X_test_df = pd.DataFrame({
            "b_t": [b_t],
            "b_t_1": [b_t_1],
            "a_t_lag": [a_t_lag],
            "a_t_lag_weight": [a_t_lag_weight],
            "max_corr": [corr],
            "best_lag": [float(lag)]
        })

        X_test_df[cols_to_scale] = np.log1p(X_test_df[cols_to_scale])
        X_test_df[cols_to_scale] = x_scaler.transform(X_test_df[cols_to_scale])
        y_pred_scaled = reg.predict(X_test_df)[0]
        y_pred_log = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]
        y_pred_raw = np.expm1(y_pred_log)

        y_pred = max(0.0, float(y_pred_raw))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
            "max_corr": corr,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred





