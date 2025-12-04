# main.py
import pandas as pd

from automl import automl
from data import data_load, data_preparing, build_training_data
from util import find_comovement_pairs, baseline
from model import get_xgb_model, get_cat_model, get_extra_model
from train import fit, predict_ensemble, predict

def main():
    print("="*80)
    print("data loading...")
    data = data_load()
    print("data loading complete!")
    print("=" * 80)

    monthly, pivot_df_value,pivot_df_weight, pivot_value_smooth, pivot_weight_smooth = data_preparing(data)
    print(pivot_df_value)
    print("=" * 80)
    print("find comovement pairs")
    print("=" * 80)
    pairs = find_comovement_pairs(
        pivot_df_value
    )
    print(pairs)
    print("=" * 80)
    print(f" 탐색한 공행성쌍 수: {len(pairs)}")
    print("=" * 80)
    print("build training data")
    print("=" * 80)
    df_train = build_training_data(
        pivot_df_value,
        pairs
    )
    print(df_train)
    #모델 두 개 생성
    print("=" * 80)
    print("create models")
    print("=" * 80)
    model_xgb = get_xgb_model()
    model_extra = get_extra_model()
    model_cat = get_cat_model()
    print("=" * 80)
    print("Fitting XGBoost...")
    print("=" * 80)
    model_xgb = fit(df_train, model_xgb)
    print("=" * 80)
    print("Fitting extra tree...")
    print("=" * 80)
    model_extra = fit(df_train, model_extra)
    print("=" * 80)
    print("Fitting CatBoost...")
    print("=" * 80)
    model_cat = fit(df_train, model_cat)
    print("model fit complete!")
    print("=" * 80)

    #앙상블 예측
    print("=" * 80)
    print("Ensemble predicting...")
    print("=" * 80)
    submission = predict_ensemble(
        pivot_df_value,
        pairs,
        model_xgb,
        model_extra,
        model_cat,
        w_xgb=0.3,  # XGBoost 가중치
        w_extra=0.2, #ExtraTree 가중치
        w_cat=0.5   # CatBoost 가중치
    )

    print("=" * 80)
    print("model predict complete!")
    print("=" * 80)
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