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
    #pivot_df_value 데이터 스무딩한 피벗 데이터
    pivot_value_smooth = pivot_df_value.rolling(window=3, axis=1).mean().fillna(0)
    #pivot_df_weight 데이터 스무딩한 피벗 데이터
    pivot_weight_smooth = pivot_df_weight.rolling(window=3, axis=1).mean().fillna(0)


    return monthly, pivot_df_value, pivot_df_weight, pivot_weight_smooth, pivot_value_smooth


def build_training_data(
    pivot_value,
    pivot_value_smooth,
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

        a_v = pivot_value.loc[leader].values.astype(float)
        a_vs = pivot_value_smooth.loc[leader].values.astype(float)
        b_v = pivot_value.loc[follower].values.astype(float)
        b_vs = pivot_value_smooth.loc[follower].values.astype(float)


        # t+1이 존재하고, t-lag >= 0인 구간만 학습에 사용
        for t in range(max(lag, 1), n_months - 1):
            b_t = b_v[t]
            b_t_1 = b_v[t - 1]
            a_t_lag = a_v[t - lag]
            a_t_lag_smooth_value = a_vs[t - lag]
            b_t_smooth_value = b_vs[t]
            max_corr = corr
            best_lag = float(lag)

            rows.append({
                # value series feature
                "b_t": b_t,
                "b_t_1": b_t_1,
                "a_t_lag": a_t_lag,

             # smooth value feature
                #"a_t_lag_smooth_value": a_t_lag_smooth_value,
                #"b_t_smooth_value": b_t_smooth_value,

                # correlation info
                "max_corr": max_corr,
                "best_lag": best_lag,

                # target
                "target": b_v[t + 1]
            })

    df_train = pd.DataFrame(rows)
    return df_train