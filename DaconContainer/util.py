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
    pivot_weight,
    pivot_value_smooth,
    pivot_weight_smooth,
    max_lag=14,
    min_nonzero=12,
    value_threshold=0.37,   # 1차 value 필터
    final_threshold=0.4    # 최종 필터
):

    items = pivot_value.index.to_list()
    n_months = pivot_value.shape[1]
    results = []

    for leader in tqdm(items, desc="Finding Pairs(v3)"):

        # ----- leader series -----
        x_v = pivot_value.loc[leader].values.astype(float)
        x_w = pivot_weight.loc[leader].values.astype(float)
        x_sv = pivot_value_smooth.loc[leader].values.astype(float)
        x_sw = pivot_weight_smooth.loc[leader].values.astype(float)

        # unit price (안전 처리)
        x_up = np.divide(x_v, x_w, out=np.zeros_like(x_v), where=(x_w != 0))

        # log-diff (diff 길이 앞에서 처리)
        x_dv = np.diff(np.log1p(x_v))
        x_dw = np.diff(np.log1p(x_w))

        if np.count_nonzero(x_v) < min_nonzero:
            continue

        for follower in items:

            if follower == leader:
                continue

            # ----- follower series -----
            y_v = pivot_value.loc[follower].values.astype(float)
            y_w = pivot_weight.loc[follower].values.astype(float)
            y_sv = pivot_value_smooth.loc[follower].values.astype(float)
            y_sw = pivot_weight_smooth.loc[follower].values.astype(float)

            y_up = np.divide(y_v, y_w, out=np.zeros_like(y_v), where=(y_w != 0))

            y_dv = np.diff(np.log1p(y_v))
            y_dw = np.diff(np.log1p(y_w))

            if np.count_nonzero(y_v) < min_nonzero:
                continue

            best_corr = 0.0
            best_lag = None
            best_source = None

            # ================================
            #     LAG 탐색
            # ================================
            for lag in range(1, max_lag + 1):

                if lag >= n_months:
                    continue

                # ------------------------------
                # 1단계: value correlation 먼저 확인
                # ------------------------------
                corr_value = safe_corr(x_v[:-lag], y_v[lag:])
                if abs(corr_value) < value_threshold:
                    continue    # 1차 필터에서 탈락

                # ------------------------------
                # 2단계: full corr 계산 (quality filtering)
                # ------------------------------
                corr_list = [
                    ("vv", corr_value),
                    ("ww", safe_corr(x_w[:-lag], y_w[lag:])),
                    ("up", safe_corr(x_up[:-lag], y_up[lag:])),
                    ("svv", safe_corr(x_sv[:-lag], y_sv[lag:])),
                    ("sww", safe_corr(x_sw[:-lag], y_sw[lag:])),
                ]

                # log-diff는 길이가 n-1이므로 조정
                if len(x_dv) > lag and len(y_dv) > lag:
                    corr_list.append(("dv", safe_corr(x_dv[:-lag], y_dv[lag:])))
                if len(x_dw) > lag and len(y_dw) > lag:
                    corr_list.append(("dw", safe_corr(x_dw[:-lag], y_dw[lag:])))

                # ------------------------------
                # best correlation 선택
                # ------------------------------
                for source, corr in corr_list:
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                        best_source = source

            # ================================
            #     3단계 : final threshold 적용
            # ================================
            if best_lag is not None and abs(best_corr) >= final_threshold:
                results.append({
                    "leading_item_id": leader,
                    "following_item_id": follower,
                    "best_lag": best_lag,
                    "max_corr": best_corr,
                    "source": best_source,
                })

    # 정렬 후 중복 제거
    pairs = (
        pd.DataFrame(results)
        .assign(abs_corr=lambda df: df["max_corr"].abs())
        .sort_values("abs_corr", ascending=False)
        .drop_duplicates(["leading_item_id", "following_item_id"])
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