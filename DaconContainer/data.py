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
        .groupby(["item_id", "year", "month"], as_index=False)[["value"]]
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

    pivot_smooth_value = pivot_value.T.rolling(window=3).mean().T.fillna(0)
    pivot_smooth_6 = pivot_value.T.rolling(window=6).mean().T.fillna(0)
    pivot_std_3 = pivot_value.T.rolling(window=3).std().T.fillna(0)

    return monthly, pivot_value, pivot_smooth_value, pivot_std_3, pivot_smooth_6

def build_training_data(
        pivot_value,
        pivot_smooth_value,
        pivot_std_3,
        pivot_smooth_6,
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
        a_val = pivot_value.loc[leader].values
        a_sm3 = pivot_smooth_value.loc[leader].values
        a_std3 = pivot_std_3.loc[leader].values
        a_sm6 = pivot_smooth_6.loc[leader].values
        b_series = pivot_value.loc[follower].values.astype(float)

        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            b_t = b_series[t]
            b_t_1 = b_series[t - 1]
            a_t_lag = a_series[t - lag]
            b_t_plus_1 = b_series[t + 1]

            idx = t + 1 - lag
            if idx < 0: continue

            val_lag = a_val[idx]  # 원본 값
            sm3_lag = a_sm3[idx]  # 3개월 평균
            std3_lag = a_std3[idx]  # 3개월 변동성
            sm6_lag = a_sm6[idx]  # 6개월 평균

            disparity = val_lag / (sm3_lag + 1)

            # 2. 골든크로스 신호: 단기 추세가 장기 추세보다 높은가?
            trend_strength = sm3_lag / (sm6_lag + 1)

            rows.append({
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,
                "max_corr": corr,
                "best_lag": float(lag),
                "a_smooth_3": sm3_lag,      # 안정적 추세
                "a_std_3": std3_lag,  # 리스크(변동성)
                "a_disparity": disparity,  # 괴리율 (스케일링 효과)
                "a_trend_strength": trend_strength,  # 상승/하락 강도
                "target": np.log1p(b_t_plus_1),
            })

    df_train = pd.DataFrame(rows)
    return df_train
