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


def fit(hard_voting_model):
    df_train_model = create_train()

    feature_cols = ['b_t', 'b_t_1', 'a_t_lag','a_t_lag_weight', 'max_corr', 'best_lag']

    train_X = df_train_model[feature_cols].values
    train_y = df_train_model["target"].values

    hard_voting_model.fit(train_X, train_y)

    return hard_voting_model


def predict(pivot_df_value,pivot_df_weight, pairs, reg):
    months = pivot_df_value.columns.to_list()
    n_months = len(months)

    # 가장 마지막 두 달 index (2025-7, 2025-6)
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

        # t_last - lag 가 0 이상인 경우만 예측
        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]
        a_t_lag_weight = a_weight_series[t_last - lag]

        X_test = np.array([[b_t, b_t_1, a_t_lag, a_t_lag_weight, corr, float(lag)]])
        y_pred = reg.predict(X_test)[0]

        # (후처리 1) 음수 예측 → 0으로 변환
        # (후처리 2) 소수점 → 정수 변환 (무역량은 정수 단위)
        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred





