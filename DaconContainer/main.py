#main.py
# main.py (최종 수정본)

# main.py (HS4 추가 수정본)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from data import data_load, data_preparing, find_comovement_pairs, build_training_data, tranfrom_log_minmax
from EDA import EDA_run
from automl import automl1, automl2
from train import predict
from util import baseline


def main():
    train = data_load()
    print(train.head())

    # ★ 1. (신규) HS4 매핑 테이블 생성 (data_preparing보다 먼저!)
    #    'hs4_code'가 train.csv에 있는 컬럼명이라고 가정합니다.
    hs_col_name = 'hs4'  # 만약 컬럼명이 다르면 이 부분만 수정

    if hs_col_name not in train.columns:
        print(f"오류: train.csv에 '{hs_col_name}' 컬럼이 없습니다.")
        print("HS4 피처 없이 계속 진행합니다.")
        item_hs4_map = {}  # 빈 맵
    else:
        print("HS4 매핑 테이블 생성 중...")
        # item_id와 hs4_code의 고유한 조합을 딕셔너리로 만듭니다.
        item_hs4 = train[['item_id', hs_col_name]].drop_duplicates().set_index('item_id')
        item_hs4_map = item_hs4[hs_col_name].to_dict()
        print("HS4 매핑 테이블 생성 완료.")

    print("=====data preparing=====")
    monthly, pivot_df_value, pivot_df_weight = data_preparing(train)

    # --- 2. 후보군 생성 ---
    print("=====후보 공행성쌍 탐색 (낮은 임계값)=====")
    pairs_value = find_comovement_pairs(pivot_df_value, pivot_df_value, corr_threshold=0.0)
    pairs_weight = find_comovement_pairs(pivot_df_weight, pivot_df_value, corr_threshold=0.0)

    # ... (all_pairs 중복 제거 로직은 동일) ...
    all_pairs_raw = pd.concat([pairs_value, pairs_weight])
    all_pairs_raw['abs_corr'] = all_pairs_raw['max_corr'].abs()
    all_pairs_sorted = all_pairs_raw.sort_values(by='abs_corr', ascending=False)
    all_pairs = all_pairs_sorted.drop_duplicates(
        subset=["leading_item_id", "following_item_id"],
        keep="first"
    )

    # ★ 3. (신규) all_pairs에 HS4 정보 병합 (merge 대신 map 사용)
    if item_hs4_map:  # 맵이 비어있지 않다면
        print("all_pairs에 HS4 정보 병합 중...")
        all_pairs['leader_hs4'] = all_pairs['leading_item_id'].map(item_hs4_map).fillna('UNKNOWN')
        all_pairs['follower_hs4'] = all_pairs['following_item_id'].map(item_hs4_map).fillna('UNKNOWN')
        print("HS4 정보 병합 완료.")
    else:
        # HS4 컬럼이 없으면, build_training_data가 오류나지 않도록 빈 컬럼 추가
        all_pairs['leader_hs4'] = 'UNKNOWN'
        all_pairs['follower_hs4'] = 'UNKNOWN'

    # ... (EDA 부분) ...

    # --- 4. 피처 엔지니어링 ---
    # build_training_data는 HS4 정보가 추가된 all_pairs를 전달받음
    print("=====전체 학습 데이터 생성 (피처 엔지니어링)=====")
    df_train_all = build_training_data(pivot_df_value, pivot_df_weight, all_pairs)

    # ... (이후 tranfrom_log_minmax, 데이터 분리, 모델 학습, 임계값 탐색, 예측 코드는
    #    이전에 알려드린 'main.py (최종 수정본)'과 '완전히 동일'합니다.) ...

    print("=======데이터 스케일링 완료=======\n")
    df_train_scaled_all, x_scaler, y_scaler = tranfrom_log_minmax(df_train_all)

    df_train_clf = df_train_scaled_all.copy()
    df_train_clf['target'] = (df_train_all['target'] > 0).astype(int)
    print(df_train_clf['target'].value_counts())

    df_train_reg = df_train_scaled_all[df_train_all['target'] > 0].copy()
    print(f"회귀(automl2) 학습 데이터 shape: {df_train_reg.shape}")

    print("\n=======임계값 탐색을 위한 훈련/검증 데이터 분리=======")
    X_clf = df_train_clf.drop(columns=['target'])
    y_clf = df_train_clf['target']

    X_train_for_clf, X_val_for_clf, y_train_for_clf, y_val_for_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    df_train_for_clf = X_train_for_clf.copy()
    df_train_for_clf['target'] = y_train_for_clf

    print(f"automl1 훈련 데이터: {df_train_for_clf.shape}")
    print(f"임계값 검증 데이터: {X_val_for_clf.shape}")

    print("\n=======모델 1 (분류기 F1) 학습 시작=======")
    model_clf = automl1(df_train_for_clf)
    print("=======모델 1 (분류기 F1) 학습 완료=======\n")

    print("=======분류기(automl1) 최적 임계값 탐색 시작=======")
    y_pred_proba = model_clf.predict_proba(X_val_for_clf)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.1, 0.9, 0.05)

    for th in thresholds:
        y_pred_binary = (y_pred_proba > th).astype(int)
        score = f1_score(y_val_for_clf, y_pred_binary)
        print(f"Threshold: {th:.2f}, F1 Score: {score:.6f}")

        if score > best_f1:
            best_f1 = score
            best_threshold = th

    print(f"=======최적 임계값 탐색 완료=======")
    print(f"★ 최적 임계값: {best_threshold:.2f} (검증 F1: {best_f1:.6f})\n")

    print("=======모델 2 (회귀 MAE) 학습 시작=======")
    model_reg = automl2(df_train_reg)
    print("=======모델 2 (회귀 MAE) 학습 완료=======\n")

    print("=======예측 시작=======")
    submission = predict(
        pivot_df_value=pivot_df_value,
        pivot_df_weight=pivot_df_weight,
        pairs=all_pairs,
        model_clf=model_clf,
        model_reg=model_reg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        optimal_threshold=best_threshold
    )
    print("=======predict complete=======\n")
    print(submission.head())

    baseline(submission)
    print("baseline_submission.csv 생성 완료 (경로: util.py에 지정된 위치)")


if __name__ == "__main__":
    print("=======main 시작 (2-모델 + 최적 임계값 + HS4)=======🤞\n")
    main()
"""from operator import concat
import pandas as pd
from data import data_load, data_preparing, find_comovement_pairs, build_training_data, tranfrom_log_minmax
from EDA import EDA_run
from automl import automl
from model import model
from train import predict, fit
from util import baseline


def main():
    train = data_load()
    print(train.head())
    print("=====data preparing=====")
    monthly,pivot_df_value, pivot_df_weight = data_preparing(train)
    pairs_value = find_comovement_pairs(pivot_df_value,pivot_df_value)
    pairs_weight = find_comovement_pairs(pivot_df_weight,pivot_df_value)
    all_pairs = pd.concat([pairs_value, pairs_weight])

    answer = input("EDA를 진행할까요? (y/n) >>")
    if answer == "y":
        EDA_run(train)
    elif answer == "n":
        print("EDA를 건너뜀\n")

    print("탐색된 공행성쌍 수:", len(pairs_value) + len(pairs_weight))
    print("-------pairs_value-------")
    print(pairs_value.head())
    print("-------pairs_weight-------")
    print(pairs_weight.head())
    print("-------add_pairs-------")
    print(pairs_value.head())
    print("\n")

    df_train = build_training_data(pivot_df_value, pivot_df_weight, all_pairs)
    df_train, x_scaler, y_scaler = tranfrom_log_minmax(df_train)
    print(df_train)
    print("=======train_x,y split complete=======\n")
    hard_voting_model = automl(df_train)
    #hard_voting_model = model()
    #fit(hard_voting_model,df_train)
    print("=======voting model fit complete=======\n")
    submission = predict(pivot_df_value,pivot_df_weight, all_pairs, hard_voting_model,x_scaler,y_scaler)
    print("=======predict complete=======\n")
    submission.head()

    baseline(submission)
    if answer == "m":
        print("baseline_submission 생성완료 (Dacon/baselinez)")
    elif answer == "w":
        print("baseline_submission 생성완료 (Dacon/baseline)")
    elif answer == 1:
        print("baseline_submission 생성실패")


if __name__ == "__main__":
    print("=======main 시작=======🤞\n")
    main()"""
