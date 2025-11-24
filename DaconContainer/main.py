# main.py
import pandas as pd
from data import data_load, data_preparing, build_training_data
from util import find_comovement_pairs, baseline, evaluate_train
from model import get_xgb_model, get_cat_model  # [변경] 함수 이름 변경
from train import fit, predict_ensemble  # [변경] predict 대신 predict_ensemble 사용


def main():
    print("data loading...")
    data = data_load()
    print("data loading complete!")

    monthly, pivot_df_value, pivot_df_weight, pivot_value_smooth, pivot_weight_smooth = data_preparing(data)

    print("=======find comovement pairs=======")
    pairs = find_comovement_pairs(
        pivot_df_value,
        pivot_df_weight,
        pivot_value_smooth,
        pivot_weight_smooth
    )
    print(f" 탐색한 공행성쌍 수: {len(pairs)}")

    print("=======build training data=======")
    df_train = build_training_data(
        pivot_df_value,
        pairs
    )

    # 1. 모델 두 개 생성
    print("=======create models=======")
    model_xgb = get_xgb_model()
    model_cat = get_cat_model()

    # 2. 각각 학습 (Fit)
    print("Fitting XGBoost...")
    model_xgb = fit(df_train, model_xgb)

    print("Fitting CatBoost...")
    model_cat = fit(df_train, model_cat)
    print("model fit complete!")

    # 3. 앙상블 예측 (Soft Voting)
    print("Ensemble predicting...")
    submission = predict_ensemble(
        pivot_df_value,
        pairs,
        model_xgb,
        model_cat,
        w_xgb=0.4,  # XGBoost 가중치
        w_cat=0.6  # CatBoost 가중치
    )
    print("model predict complete!")

    # (참고) evaluate_train은 앙상블된 모델 하나가 아니므로
    # 여기서는 생략하거나, 개별 모델 성능만 확인해야 합니다.
    # print("=======Evaluate XGBoost Only (Reference)=======")
    # evaluate_train(df_train, model_xgb)

    answer = baseline(submission)
    if answer == "m":
        print("baseline_submission 생성완료 (Mac)")
    elif answer == "w":
        print("baseline_submission 생성완료 (Windows)")
    elif answer == 1:
        print("baseline_submission 생성실패")


if __name__ == "__main__":
    print("=======main 시작=======\n")
    main()