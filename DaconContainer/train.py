#train.py
# train.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import data_load, data_preparing, find_comovement_pairs, build_training_data


# --- 이 함수들은 현재 2-모델 전략의 main.py에서 사용되지 않습니다 ---
def create_train():
    train = data_load()
    monthly, pivot_df_value, pivot_df_weight = data_preparing(train)
    # 'corr_threshold'가 지정되지 않았으므로 data.py의 기본값(0.0)을 사용합니다.
    pairs = find_comovement_pairs(pivot_df_value)

    df_train_model = build_training_data(pivot_df_value, pivot_df_weight, pairs)
    print('생성된 학습 데이터의 shape :', df_train_model.shape)
    df_train_model.head()

    return df_train_model


def fit(model, df_train):
    # 이 feature_cols 리스트는 이제 data.py와 일치하지 않습니다. (오래된 코드)
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight', 'max_corr', 'best_lag']

    train_X = df_train[feature_cols]
    train_y = df_train["target"]

    model.fit(train_X, train_y)

    return model


# --- 여기까지 사용되지 않는 함수들 ---


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ 2-모델 전략의 핵심 예측 함수 ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def predict(pivot_df_value, pivot_df_weight, pairs, model_clf, model_reg, x_scaler, y_scaler, optimal_threshold):
    # ★ (수정) data.py의 tranfrom_log_minmax와 동일한 스케일링 대상
    cols_to_scale = [
        'b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight',
        'b_mean_3m', 'b_mean_6m', 'b_mean_12m',
        'b_std_3m', 'b_std_6m', 'b_std_12m',
        'a_mean_3m_lag', 'a_mean_12m_lag', 'a_std_12m_lag',
        'b_unit_price_t', 'a_unit_price_t_lag',
        'b_weight_mean_6m', 'b_weight_std_12m'
    ]
    # ★ FLAML은 feature_cols 리스트가 필요 없습니다.

    months = pivot_df_value.columns.to_list()
    n_months = len(months)
    t_last = n_months - 1
    t_prev = n_months - 2
    MIN_HISTORY = 12  # data.py와 동일한 최소 기간

    preds = []

    for row in tqdm(pairs.itertuples(index=False), desc="테스트 데이터 예측 중"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        # ★ (신규) HS4 정보 가져오기 (main.py가 전달해 줌)
        leader_hs4 = row.leader_hs4
        follower_hs4 = row.follower_hs4

        if leader not in pivot_df_value.index or follower not in pivot_df_value.index:
            continue
        if leader not in pivot_df_weight.index or follower not in pivot_df_weight.index:
            continue

        a_series = pivot_df_value.loc[leader].values.astype(float)
        a_weight_series = pivot_df_weight.loc[leader].values.astype(float)
        b_series = pivot_df_value.loc[follower].values.astype(float)
        b_weight_series = pivot_df_weight.loc[follower].values.astype(float)  # ★ (추가) 팔로워 무게

        # ★ (수정) 12개월치 데이터 없으면 스킵 (학습 때와 동일)
        if t_last - lag < (MIN_HISTORY - 1) or t_last < (MIN_HISTORY - 1):
            continue

        # --- 1. 기본 피처 (t_last, t_last-lag 기준) ---
        b_t = b_series[t_last]
        b_t_1 = b_series[t_prev]
        a_t_lag = a_series[t_last - lag]
        a_t_lag_weight = a_weight_series[t_last - lag]

        # --- 2. Follower(b)의 가치(Value) 시계열 피처 (t_last 기준) ---
        b_past_12m = b_series[t_last - 11: t_last + 1]  # t_last 포함, 12개
        b_features = {
            'b_mean_3m': np.mean(b_past_12m[-3:]),
            'b_mean_6m': np.mean(b_past_12m[-6:]),
            'b_mean_12m': np.mean(b_past_12m),
            'b_std_3m': np.std(b_past_12m[-3:]),
            'b_std_6m': np.std(b_past_12m[-6:]),
            'b_std_12m': np.std(b_past_12m),
            'b_zeros_12m': np.sum(b_past_12m == 0),
            'b_momentum_3_12': (np.mean(b_past_12m[-3:]) + 1e-6) / (np.mean(b_past_12m) + 1e-6)
        }

        # --- 3. Leader(a)의 가치(Value) 시계열 피처 (t_last-lag 기준) ---
        a_past_12m_lagged = a_series[t_last - lag - 11: t_last - lag + 1]  # t_last-lag 포함
        a_features = {
            'a_mean_3m_lag': np.mean(a_past_12m_lagged[-3:]),
            'a_mean_12m_lag': np.mean(a_past_12m_lagged),
            'a_std_12m_lag': np.std(a_past_12m_lagged),
            'a_zeros_12m_lag': np.sum(a_past_12m_lagged == 0)
        }

        # --- 4. 무게(Weight) 및 단가(Unit Price) 피처 ---
        b_weight_t = b_weight_series[t_last]
        b_weight_past_12m = b_weight_series[t_last - 11: t_last + 1]
        a_weight_past_12m_lagged = a_weight_series[t_last - lag - 11: t_last - lag + 1]

        weight_price_features = {
            'b_unit_price_t': (b_t + 1e-6) / (b_weight_t + 1e-6),
            'a_unit_price_t_lag': (a_t_lag + 1e-6) / (a_t_lag_weight + 1e-6),
            'b_weight_mean_6m': np.mean(b_weight_past_12m[-6:]),
            'b_weight_std_12m': np.std(b_weight_past_12m),
            'b_weight_zeros_12m': np.sum(b_weight_past_12m == 0),
            'a_weight_mean_6m_lag': np.mean(a_weight_past_12m_lagged[-6:])
        }

        # --- 5. HS4 피처 ---
        hs4_features = {
            'is_same_hs4': int(leader_hs4 == follower_hs4),
            'is_same_hs2': int(str(leader_hs4)[:2] == str(follower_hs4)[:2])
        }

        # --- 6. X_test 데이터프레임 생성 (모든 피처 포함) ---
        X_test_data = {
            "b_t": [b_t],
            "b_t_1": [b_t_1],
            "a_t_lag": [a_t_lag],
            "a_t_lag_weight": [a_t_lag_weight],
            "max_corr": [corr],
            "best_lag": [float(lag)]
        }
        X_test_data.update(b_features)
        X_test_data.update(a_features)
        X_test_data.update(weight_price_features)
        X_test_data.update(hs4_features)

        X_test_df = pd.DataFrame(X_test_data)
        X_test_df = X_test_df.fillna(0.0)  # NaN 방어

        # --- 7. X_test 스케일링 (학습 때와 동일하게) ---
        X_test_df_scaled = X_test_df.copy()

        # 스케일링 대상 컬럼이 X_test_df_scaled에 있는지 확인
        cols_to_scale_test = [col for col in cols_to_scale if col in X_test_df_scaled.columns]

        X_test_df_scaled[cols_to_scale_test] = np.log1p(X_test_df_scaled[cols_to_scale_test])
        X_test_df_scaled[cols_to_scale_test] = x_scaler.transform(X_test_df_scaled[cols_to_scale_test])

        # --- 8. (★핵심★) 2-모델 예측 ---

        # 1. 분류기(model_clf)로 '거래 확률' 예측
        trade_probability = model_clf.predict_proba(X_test_df_scaled)[0, 1]

        final_value = 0

        # 2. '확률'이 '최적 임계값'보다 클 때만 회귀 모델 실행
        if trade_probability > optimal_threshold:
            y_pred_scaled = model_reg.predict(X_test_df_scaled)[0]

            # 3. 역변환
            y_pred_log = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]
            y_pred_raw = np.expm1(y_pred_log)
            y_pred = max(0.0, float(y_pred_raw))
            final_value = int(round(y_pred))

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": final_value,
            "max_corr": corr,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred


# --- 이 코드는 이전 1-모델 전략의 코드이므로 주석 처리합니다 ---
"""def predict(pivot_df_value, pivot_df_weight, pairs, reg, x_scaler, y_scaler):

    cols_to_scale = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight']

    months = pivot_df_value.columns.to_list()
... (이하 주석 처리된 코드) ...
"""
"""import numpy as np
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


def predict(pivot_df_value, pivot_df_weight, pairs, model_clf, model_reg, x_scaler, y_scaler, optimal_threshold):
    cols_to_scale = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight']
    cols_to_scale = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight']
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight', 'max_corr', 'best_lag']

    months = pivot_df_value.columns.to_list()
    n_months = len(months)
    t_last = n_months - 1
    t_prev = n_months - 2

    preds = []

    # 'pairs'는 이제 'corr_threshold=0.4'로 필터링된 리스트가 아닌
    # 'corr_threshold=0.1'로 생성된 '모든 후보' 리스트임
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

        # X_test 데이터프레임 생성
        X_test_df = pd.DataFrame({
            "b_t": [b_t],
            "b_t_1": [b_t_1],
            "a_t_lag": [a_t_lag],
            "a_t_lag_weight": [a_t_lag_weight],
            "max_corr": [corr],
            "best_lag": [float(lag)]
        })

        # X_test 스케일링 (학습 때와 동일하게)
        X_test_df_scaled = X_test_df.copy()  # 원본 피처 보존 (FLAML이 내부 스케일링 안 썼을 경우 대비)
        X_test_df_scaled[cols_to_scale] = np.log1p(X_test_df_scaled[cols_to_scale])
        X_test_df_scaled[cols_to_scale] = x_scaler.transform(X_test_df_scaled[cols_to_scale])
        trade_probability = model_clf.predict_proba(X_test_df_scaled[feature_cols])[0, 1]

        # ★ Key Change: 2-모델 예측

        # 1. 분류기(model_clf)로 거래 여부(0/1) 예측
        #    FLAML은 스케일링된 X(feature_cols 순서)를 기대함
        final_value = 0  # 기본값 0

        if trade_probability > optimal_threshold:
            # 2. 분류기가 '거래함(1)'이라고 예측한 경우에만 회귀기(model_reg)로 거래량 예측
            y_pred_scaled = model_reg.predict(X_test_df_scaled[feature_cols])[0]

            # ... (역변환 코드 동일) ...
            y_pred_log = y_scaler.inverse_transform([[y_pred_scaled]])[0][0]
            y_pred_raw = np.expm1(y_pred_log)  # log1p 역변환
            y_pred = max(0.0, float(y_pred_raw))
            final_value = int(round(y_pred))  # 평가 산식에 따라 정수 반올림

        preds.append({
            "leading_item_id": leader,
            "following_item_id": follower,
            "value": final_value,  # 0 또는 예측된 거래량
            "max_corr": corr,
        })

    df_pred = pd.DataFrame(preds)
    return df_pred

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
    return df_pred"""





