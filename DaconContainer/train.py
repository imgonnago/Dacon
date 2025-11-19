#train.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from data import data_load, data_preparing, find_comovement_pairs, build_training_data


def create_train():
    train = data_load()
    monthly, pivot_df_value, pivot_df_weight, pivot_weight_smooth, pivot_value_smooth = data_preparing(train)

    # hs4 매핑
    item_to_hs4_map = train.set_index("item_id")["hs4"].to_dict()

    # main.py의 pair 구조 그대로 반영
    pairs_value         = find_comovement_pairs(pivot_df_value, pivot_df_value)
    pairs_weight        = find_comovement_pairs(pivot_df_weight, pivot_df_value)
    pairs_value_smooth  = find_comovement_pairs(pivot_value_smooth,  pivot_value_smooth)
    pairs_weight_smooth = find_comovement_pairs(pivot_weight_smooth, pivot_value_smooth)

    all_pairs = pd.concat([
        pairs_value,
        pairs_weight,
        pairs_value_smooth,
        pairs_weight_smooth
    ]).drop_duplicates(subset=['leading_item_id', 'following_item_id'])

    # build_training_data 호출 (data.py 최신 구조)
    df_train_model = build_training_data(
        item_to_hs4_map,
        pivot_df_value,
        pivot_df_weight,
        pivot_value_smooth,
        pivot_weight_smooth,
        all_pairs
    )

    print('train shape:', df_train_model.shape)
    return df_train_model


def fit(model,df_train):
    # target encoding map 생성
    te_maps = {}
    for col in ["leading_hs4", "following_hs4"]:
        te_maps[col] = df_train.groupby(col)["target"].mean()

    # 인코딩 적용
    for col in ["leading_hs4", "following_hs4"]:
        df_train[col] = df_train[col].map(te_maps[col])

    feature_cols = [
        'b_t', 'b_t_1',
        'a_t_lag', 'a_t_lag_weight',
        'a_t_lag_smooth', 'a_t_lag_weight_smooth',
        'max_corr', 'best_lag',
        'leading_hs4', 'following_hs4'
    ]

    train_X = df_train[feature_cols]
    train_y = df_train["target"]

    model.fit(train_X, train_y)

    return model, te_maps


def predict(pivot_df_value, pivot_df_weight, pairs,
            pivot_val_smooth, pivot_wgt_smooth, reg, te_mmaps):
        # target encoding 일관성 유지
    for col in ["leading_hs4", "following_hs4"]:
        X_test_df[col] = X_test_df[col].map(te_maps[col]).fillna(te_maps[col].mean())

    cols_to_scale = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight',
                     'a_t_lag_smooth', 'a_t_lag_weight_smooth']

    months = pivot_df_value.columns.to_list()
    n_months = len(months)
    t_last = n_months - 1
    t_prev = n_months - 2

    preds = []

    leaders = pairs['leading_item_id'].values
    followers = pairs['following_item_id'].values
    lags = pairs['best_lag'].values.astype(int)
    corrs = pairs['max_corr'].values.astype(float)

    for i in tqdm(range(len(pairs))):
        leader = leaders[i]
        follower = followers[i]
        lag = lags[i]
        corr = corrs[i]
        l_hs4 = item_to_hs4_map.get(leader)
        f_hs4 = item_to_hs4_map.get(follower)

        if (leader not in pivot_df_value.index or follower not in pivot_df_value.index or
                leader not in pivot_val_smooth.index or follower not in pivot_val_smooth.index or
                leader not in pivot_df_weight.index or leader not in pivot_wgt_smooth.index):
            continue

        a_series = pivot_df_value.loc[leader].values.astype(float)
        b_series = pivot_df_value.loc[follower].values.astype(float)
        a_weight_series = pivot_df_weight.loc[leader].values.astype(float)
        a_series_smooth = pivot_val_smooth.loc[leader].values.astype(float)
        a_weight_series_smooth = pivot_wgt_smooth.loc[leader].values.astype(float)

        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]
        a_t_lag_weight = a_weight_series[t_last - lag]
        a_t_lag_smooth = a_series_smooth[t_last - lag]
        a_t_lag_weight_smooth = a_weight_series_smooth[t_last - lag]

        X_test_df = pd.DataFrame({
            "b_t": [b_t],
            "b_t_1": [b_t_1],
            "a_t_lag": [a_t_lag],
            "a_t_lag_weight": [a_t_lag_weight],
            "a_t_lag_smooth": [a_t_lag_smooth],
            "a_t_lag_weight_smooth": [a_t_lag_weight_smooth],
            "max_corr": [corr],
            "best_lag": [float(lag)],
            "leading_hs4": [l_hs4],
            "following_hs4": [f_hs4]
        })


        X_test_df[cols_to_scale] = np.log1p(X_test_df[cols_to_scale])

        y_pred_scaled = reg.predict(X_test_df)[0]

        y_pred_raw = np.expm1(y_pred_scaled)

        y_pred = max(0.0, float(y_pred_raw))
        y_pred = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": y_pred,
            "max_corr": corr,
        })
        if len(preds) == 0:
            print("⚠️ 경고: 예측된 결과가 0건입니다! (데이터 매칭 실패)")
            return pd.DataFrame(columns=['leading_item_id', 'following_item_id', 'value', 'max_corr'])

    df_pred = pd.DataFrame(preds)
    return df_pred





