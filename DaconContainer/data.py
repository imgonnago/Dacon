#data.py
# data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm


def data_load():
    url = "https://raw.githubusercontent.com/imgonnago/Dacon/refs/heads/main/ACD2-Week12-1/dataset/train.csv"
    train = pd.read_csv(url)
    train.head()
    return train


# 로그변환 정규화 진행 ->  build_training_data 리턴값 사용
def tranfrom_log_minmax(train):
    # ★ 1. feature_cols에 'target'을 제외한 모든 컬럼을 넣습니다.
    feature_cols = [col for col in train.columns if col not in ['target']]

    X_train_df = train[feature_cols].copy()
    y_train_df = train[['target']].copy()

    # ★ 2. 거래량/무게/단가 기반의 피처들만 스케일링 대상으로 지정합니다.
    # (카운트, 비율, 이미 스케일링된 corr 등은 제외)
    cols_to_scale = [
        'b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight',
        'b_mean_3m', 'b_mean_6m', 'b_mean_12m',
        'b_std_3m', 'b_std_6m', 'b_std_12m',
        'a_mean_3m_lag', 'a_mean_12m_lag', 'a_std_12m_lag',
        'b_unit_price_t', 'a_unit_price_t_lag',
        'b_weight_mean_6m', 'b_weight_std_12m'
    ]

    # ★ 3. 스케일링 대상 컬럼이 X_train_df에 있는지 확인 (방어 코드)
    #    (데이터가 적어 12m 피처 등이 생성 안 될 경우 대비)
    cols_to_scale = [col for col in cols_to_scale if col in X_train_df.columns]

    X_train_df[cols_to_scale] = np.log1p(X_train_df[cols_to_scale])
    x_scaler = MinMaxScaler()
    X_train_df[cols_to_scale] = x_scaler.fit_transform(X_train_df[cols_to_scale])

    # --- y 스케일링 (Target) ---
    y_train_df['target'] = np.log1p(y_train_df['target'])
    y_scaler = MinMaxScaler()
    y_train_df['target'] = y_scaler.fit_transform(y_train_df)

    # ★ 4. X_train_df와 y_train_df를 합쳐서 반환
    train_scaled = pd.concat([X_train_df, y_train_df], axis=1)

    return train_scaled, x_scaler, y_scaler


# pivot weight, value, monthly 생성
def data_preparing(train):
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)[["value", "weight"]]
        .sum()
    )
    # year, month를 하나의 키(ym)로 묶기
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    # item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
    pivot_df_value = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )

    pivot_df_weight = (
        monthly
        .pivot(index="item_id", columns="ym", values="weight")
        .fillna(0.0)
    )

    return monthly, pivot_df_value, pivot_df_weight


# 공행성쌍 찾을때 사용
def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# 공생성쌍 탐색 -> pairs리턴
def find_comovement_pairs(df_lead, df_follow, max_lag=12, min_nonzero=12, corr_threshold=0.0):
    # ★ (수정) corr_threshold 기본값을 0.0으로 하여 모든 후보 탐색
    lead = df_lead.index.to_list()
    follow = df_follow.index.to_list()
    n_months = len(df_lead.columns)

    results = []

    for i, leader in tqdm(enumerate(lead), desc="공행성쌍 후보 탐색 중"):
        x = df_lead.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in follow:
            if follower == leader:
                continue

            y = df_follow.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                corr = safe_corr(x[:-lag], y[lag:])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            # 임계값 이상이면 공행성쌍으로 채택
            if best_lag is not None and abs(best_corr) >= corr_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                })

    pairs = pd.DataFrame(results)
    return pairs


# 학습데이터를 생성 -> 학습데이터에 weight, value를 로그, 정규화 해아햠
def build_training_data(pivot_df_value, pivot_df_weight, pairs):
    """
    (피처 엔지니어링 강화 버전)
    """
    months = pivot_df_value.columns.to_list()
    n_months = len(months)

    rows = []

    # 롤링 계산을 위한 최소 기간 (12개월치가 필요함)
    MIN_HISTORY = 12

    for row in tqdm(pairs.itertuples(index=False), desc="피처 엔지니어링 진행 중"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)
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

        # t+1(미래)이 존재하고, t-lag(과거) 및 12개월 롤링이 가능한 구간만 학습
        start_t = max(lag, 1, MIN_HISTORY - 1)

        for t in range(start_t, n_months - 1):

            # --- 1. 기본 피처 (t, t-lag 기준) ---
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            a_t_lag_weight = a_weight_series[t - lag]

            # --- 2. 정답 (Target) ---
            b_t_plus_1 = b_series[t + 1]

            # --- 3. Follower(b)의 가치(Value) 시계열 피처 (t 기준) ---
            b_past_12m = b_series[t - 11: t + 1]  # t포함, 12개

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

            # --- 4. Leader(a)의 가치(Value) 시계열 피처 (t-lag 기준) ---
            if t - lag < (MIN_HISTORY - 1): continue  # 12개월치 데이터 없으면 스킵

            a_past_12m_lagged = a_series[t - lag - 11: t - lag + 1]  # t-lag 포함, 12개

            a_features = {
                'a_mean_3m_lag': np.mean(a_past_12m_lagged[-3:]),
                'a_mean_12m_lag': np.mean(a_past_12m_lagged),
                'a_std_12m_lag': np.std(a_past_12m_lagged),
                'a_zeros_12m_lag': np.sum(a_past_12m_lagged == 0)
            }

            # --- 5. (신규) 무게(Weight) 및 단가(Unit Price) 피처 ---
            b_weight_t = b_weight_series[t]
            b_weight_past_12m = b_weight_series[t - 11: t + 1]
            a_weight_past_12m_lagged = a_weight_series[t - lag - 11: t - lag + 1]

            weight_price_features = {
                'b_unit_price_t': (b_t + 1e-6) / (b_weight_t + 1e-6),
                'a_unit_price_t_lag': (a_t_lag + 1e-6) / (a_t_lag_weight + 1e-6),
                'b_weight_mean_6m': np.mean(b_weight_past_12m[-6:]),
                'b_weight_std_12m': np.std(b_weight_past_12m),
                'b_weight_zeros_12m': np.sum(b_weight_past_12m == 0),
                'a_weight_mean_6m_lag': np.mean(a_weight_past_12m_lagged[-6:])
            }
            hs4_features = {
                # HS4 코드가 같은가? (가장 강력한 피처)
                'is_same_hs4': int(leader_hs4 == follower_hs4),
                # HS2(대분류)가 같은가? (HS4 앞 2자리)
                'is_same_hs2': int(str(leader_hs4)[:2] == str(follower_hs4)[:2])
            }

            # --- 6. 데이터 취합 ---
            row_data = {
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "a_t_lag_weight": a_t_lag_weight,
                "max_corr": corr,
                "best_lag": float(lag),
                "target": b_t_plus_1,

            }

            row_data.update(b_features)
            row_data.update(a_features)
            row_data.update(weight_price_features)
            row_data.update(hs4_features)
            rows.append(row_data)

    df_train = pd.DataFrame(rows)
    df_train = df_train.fillna(0.0)

    return df_train
"""import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

def data_load():
    url = "https://raw.githubusercontent.com/imgonnago/Dacon/refs/heads/main/ACD2-Week12-1/dataset/train.csv"
    train = pd.read_csv(url)
    train.head()
    return train

#로그변환 정규화 진행 ->  build_training_data 리턴값 사용
def tranfrom_log_minmax(train):
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight', 'max_corr', 'best_lag']
    X_train_df = train[feature_cols].copy()
    y_train_df = train[['target']].copy()

    cols_to_scale = ['b_t', 'b_t_1', 'a_t_lag', 'a_t_lag_weight']

    X_train_df[cols_to_scale] = np.log1p(X_train_df[cols_to_scale])
    x_scaler = MinMaxScaler()
    X_train_df[cols_to_scale] = x_scaler.fit_transform(X_train_df[cols_to_scale])

    y_train_df['target'] = np.log1p(y_train_df['target'])
    y_scaler = MinMaxScaler()
    y_train_df['target'] = y_scaler.fit_transform(y_train_df)

    train = pd.concat([X_train_df, y_train_df], axis=1)
    return  train, x_scaler, y_scaler

#pivot weight, value, monthly 생성
def data_preparing(train):
    monthly = (
         train
        .groupby(["item_id", "year", "month"], as_index=False)[["value","weight"]]
        .sum()
    )
    # year, month를 하나의 키(ym)로 묶기
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    # item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
    pivot_df_value = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )

    pivot_df_weight = (
        monthly
        .pivot(index="item_id", columns="ym", values="weight")
        .fillna(0.0)
    )

    return monthly, pivot_df_value, pivot_df_weight

#공행성쌍 찾을때 사용
def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

#공생성쌍 탐색 -> pairs리턴
def find_comovement_pairs(df_lead, df_follow, max_lag=12, min_nonzero=12, corr_threshold=0.5):
    lead = df_lead.index.to_list()
    follow = df_follow.index.to_list()
    n_months = len(df_lead.columns)

    results = []

    for i, leader in tqdm(enumerate(lead)):
        x = df_lead.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in follow:
            if follower == leader:
                continue

            y = df_follow.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue

            best_lag = None
            best_corr = 0.0

            # lag = 1 ~ max_lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue
                corr = safe_corr(x[:-lag], y[lag:])
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

            # 임계값 이상이면 공행성쌍으로 채택
            if best_lag is not None and abs(best_corr) >= corr_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                })

    pairs = pd.DataFrame(results)
    return pairs

#학습데이터를 생성 -> 학습데이터에 weight, value를 로그, 정규화 해아햠
def build_training_data(pivot_df_value,pivot_df_weight, pairs):
   
    공행성쌍 + 시계열을 이용해 (X, y) 학습 데이터를 만드는 함수
    input X:
      - b_t, b_t_1, a_t_lag, max_corr, best_lag
    target y:
      - b_t_plus_1

    months = pivot_df_value.columns.to_list()
    n_months = len(months)

    rows = []

    for row in pairs.itertuples(index=False):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot_df_value.index or follower not in pivot_df_value.index:
            continue

        a_series = pivot_df_value.loc[leader].values.astype(float)
        a_weight_series = pivot_df_weight.loc[leader].values.astype(float)
        b_series = pivot_df_value.loc[follower].values.astype(float)

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            a_t_lag_weight = a_weight_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "a_t_lag_weight": a_t_lag_weight,
                "max_corr": corr,
                "best_lag": float(lag),
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train"""


#train[(train["item_id"]=="DBWLZWNK") & (train["year"] == 2023) & (train["month"] == 1)].value_counts()
#monthly[(monthly["item_id"]=="DBWLZWNK") & (monthly["year"] == 2023) & (monthly["month"] == 1)].value_counts()