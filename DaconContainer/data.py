#data.py
import pandas as pd
import numpy as np
from tqdm import tqdm


def data_load():
    url = "https://raw.githubusercontent.com/imgonnago/Dacon/refs/heads/main/ACD2-Week12-1/dataset/train.csv"
    train = pd.read_csv(url)
    train.head()
    return train


def data_preparing(train):
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)["value"]
        .sum()
    )
    # year, month를 하나의 키(ym)로 묶기
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    # item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
    pivot_df = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )
    return monthly, pivot_df


def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def find_comovement_pairs(pivot_df, max_lag=6, min_nonzero=12, corr_threshold=0.4):
    items = pivot_df.index.to_list()
    months = pivot_df.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items)):
        x = pivot_df.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            y = pivot_df.loc[follower].values.astype(float)
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


def build_training_data(pivot_df, pairs):
    """
    공행성쌍 + 시계열을 이용해 (X, y) 학습 데이터를 만드는 함수
    input X:
      - b_t, b_t_1, a_t_lag, max_corr, best_lag
    target y:
      - b_t_plus_1
    """
    months = pivot_df.columns.to_list()
    n_months = len(months)

    rows = []

    for row in pairs.itertuples(index=False):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot_df.index or follower not in pivot_df.index:
            continue

        a_series = pivot_df.loc[leader].values.astype(float)
        b_series = pivot_df.loc[follower].values.astype(float)

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "target": b_t_plus_1,
            })

    df_train = pd.DataFrame(rows)
    return df_train




