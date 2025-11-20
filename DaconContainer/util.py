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

def find_comovement_pairs(pivot_value,
    pivot_weight,
    pivot_value_smooth,
    pivot_weight_smooth, max_lag=12, min_nonzero=12, corr_threshold=0.6):

    items = pivot_value.index.to_list()
    months = pivot_value.columns.to_list()
    n_months = len(months)

    results = []

    for leader in tqdm(items,desc="Finding Pairs"):
        x_v = pivot_value.loc[leader].values.astype(float)
        x_w = pivot_weight.loc[leader].values.astype(float)
        x_vs = pivot_value_smooth.loc[leader].values.astype(float)
        x_ws = pivot_weight_smooth.loc[leader].values.astype(float)

        # 최소 데이터 개수 체크
        if np.count_nonzero(x_v) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            y_v = pivot_value.loc[follower].values.astype(float)
            y_w = pivot_weight.loc[follower].values.astype(float)
            y_vs = pivot_value_smooth.loc[follower].values.astype(float)
            y_ws = pivot_weight_smooth.loc[follower].values.astype(float)

            if np.count_nonzero(y_v) < min_nonzero:
                continue

            best_corr = 0.0
            best_lag = None

            # lag 탐색
            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue

                corr_v = safe_corr(x_v[:-lag], y_v[lag:])
                corr_w = safe_corr(x_w[:-lag], y_w[lag:])
                corr_vs = safe_corr(x_vs[:-lag], y_vs[lag:])
                corr_ws = safe_corr(x_ws[:-lag], y_ws[lag:])

                corr_list = [
                    ("value", corr_v),
                    ("weight", corr_w),
                    ("smooth_value", corr_vs),
                    ("smooth_weight", corr_ws),
                ]

                for source, corr in corr_list:
                    if abs(corr) >= corr_threshold:
                        results.append({
                            "leading_item_id": leader,
                            "following_item_id": follower,
                            "best_lag": lag,
                            "max_corr": corr,
                            "source": source,
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
    X_train = df_train[["b_t", "b_t_1", "a_t_lag",
                        "b_t_weight", "b_t_1_weight", "a_t_lag_weight",
                        "a_t_lag_smooth_value", "a_t_lag_smooth_weight",
                        "max_corr", "best_lag"]]
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