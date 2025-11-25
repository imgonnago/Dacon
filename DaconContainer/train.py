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

    # 가장 마지막 두 달 index (2025-7, 2025-6)
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

        # t_last - lag 가 0 이상인 경우만 예측
        if t_last - lag < 0:
            continue

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        X_test = np.array([[b_t, b_t_1, a_t_lag, corr, float(lag)]])
        y_pred = reg.predict(X_test)[0]

        # value값 log1p 에서 역변환
        y_pred = np.expm1(y_pred)

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


def predict_ensemble(pivot, pairs, model_xgb, model_cat, w_xgb = 0.4, w_cat = 0.6):
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

        # t_last - lag 계산 (예외 처리)
        if t_last - lag < 0:
            continue

        a_series = pivot.loc[leader].values.astype(float)
        b_series = pivot.loc[follower].values.astype(float)

        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]

        # 모델에 들어갈 입력값 (수치형만 사용)
        X_test = pd.DataFrame([{
            "b_t": b_t,
            "b_t_1": b_t_1,
            "a_t_lag": a_t_lag,
            "max_corr": corr,
            "best_lag": float(lag)
        }])

        # 1. 각각 예측
        pred_xgb_log = model_xgb.predict(X_test)[0]
        pred_cat_log = model_cat.predict(X_test)[0]

        # 2. 역변환 (Log -> Original)
        val_xgb = np.expm1(pred_xgb_log)
        val_cat = np.expm1(pred_cat_log)

        # 3. 소프트 보팅 (가중 평균)
        final_val = (val_xgb * w_xgb) + (val_cat * w_cat)

        # 4. 후처리 (음수 제거 및 정수 반올림)
        final_val = max(0.0, float(final_val))
        final_val = int(round(final_val))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": final_val,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred











