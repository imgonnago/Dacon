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
        submission.to_csv('/Users/joyongjae/Dacon/baseline/baseline_submission.csv', index=False)
        print("complete")
        return answer

    elif answer == "w":
        print("submission.csv is saved to window")
        submission.to_csv('C:/Users/zxfg0/Dacon/baseline/baseline_submission.csv', index=False)
        print("complete")
        return answer
    else:
        print("you didn't choose os enviroment")
        return 1

def safe_corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def find_comovement_pairs(pivot, max_lag=8, min_nonzero=10, corr_threshold=0.365):
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    for i, leader in tqdm(enumerate(items)):
        x = pivot.loc[leader].values.astype(float)
        if np.count_nonzero(x) < min_nonzero:
            continue

        for follower in items:
            if follower == leader:
                continue

            y = pivot.loc[follower].values.astype(float)
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