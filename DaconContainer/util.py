#util.py
from data import data_load
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

def baseline(submission):

    answer = input("mac/window(m/w)")
    if answer == "m":
        print("sumission.csv is saved to mac")
        submission.to_csv('/Users/joyongjae/Dacon/baseline/baseline_submit.csv', index=False)
        print("complete")
        return answer

    elif answer == "w":
        print("submission.csv is saved to window")
        submission.to_csv('C:/Users/zxfg0/Dacon/baseline/baseline_submit.csv', index=False)
        print("complete")
        return answer
    else:
        print("you didn't choose os enviroment")
        return 1

def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

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


def log1p_transform(dataset):
    return np.log1p(dataset)


def evaluate_train(df_train, model):
    """
    학습 데이터 기준 RMSE, MAE, NMAE 계산
    df_train: build_training_data로 만든 train 데이터
    model: 학습 완료된 모델
    """
    X_train = df_train[["b_t", "b_t_1", "a_t_lag", "max_corr", "best_lag"]]
    y_train = df_train["target"]

    # 예측
    y_pred_log = model.predict(X_train)      # 학습 스케일(log1p)
    y_pred = np.expm1(y_pred_log)            # 원래 스케일
    y_true = np.expm1(y_train)               # 원래 스케일

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nmae = mae / y_true.mean()

    print(f"[Train Eval] RMSE: {rmse:.2f}, MAE: {mae:.2f}, NMAE: {nmae:.4f}")
    return rmse, mae, nmae