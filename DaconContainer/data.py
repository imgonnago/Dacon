#data.py
from operator import index

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

def data_load():
    url = "https://raw.githubusercontent.com/imgonnago/Dacon/refs/heads/main/ACD2-Week12-1/dataset/train.csv"
    train = pd.read_csv(url)
    train.head()
    return train

#pivot weight, value, monthly 생성
def data_preparing(train):
    monthly = (
        train
        .groupby(["item_id", "year", "month"], as_index=False)[["value","weight"]]
        .sum()
    )
    monthly["ym"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )

    pivot_value = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )

    pivot_weight = (
        monthly
        .pivot(index="item_id", columns="ym", values="value")
        .fillna(0.0)
    )

    pivot_smooth_value = pivot_value.T.rolling(window=3).mean().T.fillna(0)
    pivot_smooth_weight = pivot_weight.T.rolling(window=3).mean().T.fillna(0)


    return monthly, pivot_value,pivot_weight, pivot_smooth_value,pivot_smooth_weight

def build_training_data(
        pivot_value,
        pairs
    ):
    """
    공행성쌍 + 시계열을 이용해 (X, y) 학습 데이터를 만드는 함수
    input X:
      - b_t, b_t_1, a_t_lag, max_corr, best_lag
    target y:
      - b_t_plus_1
    """
    months = pivot_value.columns.to_list()
    n_months = len(months)

    rows = []

    for row in tqdm(pairs.itertuples(index=False), desc="build train data"):
        leader = row.leading_item_id
        follower = row.following_item_id
        lag = int(row.best_lag)
        corr = float(row.max_corr)

        if leader not in pivot_value.index or follower not in pivot_value.index:
            continue

        a_series = pivot_value.loc[leader].values.astype(float)
        b_series = pivot_value.loc[follower].values.astype(float)

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
                "target": np.log1p(b_t_plus_1),
            })

    df_train = pd.DataFrame(rows)
    return df_train
