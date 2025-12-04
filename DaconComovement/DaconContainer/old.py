def predict(
        pivot_value,
        pivot_value_smooth,
        pairs,
        model):

    months = pivot_value.columns.to_list()
    n_months = len(months)

    # 가장 마지막 두 달 index (2025-7, 2025-6)
    t_last = n_months - 1
    t_prev = n_months - 2

    preds = []

    for row in tqdm(pairs.itertuples(index=False), desc="model predict..."):
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

        # t_last - lag 가 0 이상인 경우만 예측
        if t_last - lag < 0:
            continue

        b_t = b_v[t_last]
        b_t_1 = b_v[t_prev]
        a_t_lag = a_v[t_last - lag]
        a_t_lag_smooth_value = a_vs[t_last - lag]
        b_t_smooth_value = b_vs[t_last]
        max_corr = corr
        best_lag = float(lag)

        X_test = pd.DataFrame(
            [[b_t, b_t_1, a_t_lag, max_corr, best_lag]],
            columns=["b_t", "b_t_1", "a_t_lag", "max_corr", "best_lag"]
        )

        y_pred = model.predict(X_test)[0]

        #value값 log1p 에서 역변환
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

def find_comovement_pairs(
    pivot_value,
    pivot_weight,               # 형태는 유지하지만 사용 안함
    pivot_value_smooth,
    pivot_weight_smooth,        # 형태는 유지하지만 사용 안함
    max_lag=12,
    min_nonzero=12,
    corr_threshold=0.4):

    items = pivot_value.index.to_list()
    months = pivot_value.columns.to_list()
    n_months = len(months)

    results = []

    for leader in tqdm(items, desc="Finding Pairs"):

        x_v = pivot_value.loc[leader].values.astype(float)
        x_w = pivot_weight.loc[leader].values.astype(float)
        x_sv = pivot_value_smooth.loc[leader].values.astype(float)
        x_sw = pivot_weight_smooth.loc[leader].values.astype(float)
        diff_xv = np.diff(np.log1p(x_v))
        if np.count_nonzero(x_v) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue


            y_v = pivot_value.loc[follower].values.astype(float)
            y_w = pivot_weight.loc[follower].values.astype(float)
            y_sv = pivot_value_smooth.loc[follower].values.astype(float)
            y_sw = pivot_weight_smooth.loc[follower].values.astype(float)
            diff_yv = np.diff(np.log1p(y_v))

            if np.count_nonzero(y_v) < min_nonzero:
                continue

            best_corr = 0.0
            best_lag = None
            best_source = None

            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue

                corr_vv = safe_corr(x_v[:-lag], y_v[lag:])
                corr_wv = safe_corr(x_w[:-lag], y_v[lag:])
                corr_svv = safe_corr(x_sv[:-lag], y_sv[lag:])
                corr_swv = safe_corr(x_sw[:-lag], y_sv[lag:])
                corr_sww = safe_corr(x_sw[:-lag], y_sw[lag:])
                corr_diff = safe_corr(diff_xv[:-lag], diff_yv[lag:])

                corr_list=[
                    ("vv", corr_vv),
                    #("wv", corr_wv),
                    #("svv", corr_svv),
                    #("swv", corr_swv),
                    #("sww", corr_sww),
                    #("diffvv", corr_diff)
                ]


                for source, corr in corr_list:
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                        best_source = source

            if best_lag is not None and abs(best_corr) >= corr_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                    "source": best_source,
                })

    pairs = (
        pd.DataFrame(results)
        .assign(abs_corr=lambda df: df["max_corr"].abs())
        .sort_values(by="abs_corr", ascending=False)
        .drop_duplicates(subset=["leading_item_id", "following_item_id"], keep="first")
        .drop(columns=["abs_corr"])
    )

    return pairs


def build_training_data(
        pivot_value,
        pivot_value_smooth,
        pairs
):

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
            # "a_t_lag_smooth_value": a_t_lag_smooth_value,
            # "b_t_smooth_value": b_t_smooth_value,

            # correlation info
            "max_corr": max_corr,
            "best_lag": best_lag,

            # target
            "target": b_v[t + 1]
        })

df_train = pd.DataFrame(rows)
return df_train