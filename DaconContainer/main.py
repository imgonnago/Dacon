#main.py
import pandas as pd
from data import data_load, data_preparing,build_training_data
from util import find_comovement_pairs, log1p_transform,baseline, evaluate_train
from automl import automl
from model import model
from train import predict, fit


def main():
    print("data loading...")
    data = data_load()
    print("data loading complete!")
    monthly, pivot_df_value, pivot_df_weight, pivot_value_smooth, pivot_weight_smooth = data_preparing(data)
    print("monthly, pivot_df_value, pivot_df_weight, pivot_value_smooth, pivot_weight_smooth is created")
    print("=======df log1p transform=======")
    pivot_df_value_log = log1p_transform(pivot_df_value)
    pivot_df_weight_log = log1p_transform(pivot_df_weight)
    pivot_value_smooth_log = log1p_transform(pivot_value_smooth)
    pivot_weight_smooth_log = log1p_transform(pivot_weight_smooth)
    print("transform complete!")
    print(f"pivot_df_value_log\n{pivot_df_value_log}")
    print(f"pivot_value_smooth_log\n{pivot_value_smooth_log}")
    print("=======find comovement pairs=======")
    pairs = find_comovement_pairs(
        pivot_df_value_log,
        pivot_df_weight_log,
        pivot_value_smooth_log,
        pivot_weight_smooth_log
    )
    print(f"comovement finding complete\n{pairs}")
    print(f" 탐색한 공행성쌍 수: {len(pairs)}")
    print("=======create model=======")

    df_train = build_training_data(
        pivot_df_value_log,
        pivot_value_smooth_log,
        pairs
    )

    #Model = model()
    Model = automl(df_train)
    print("fit...")
    #Model = fit(df_train,Model)
    print("model fit complete!")
    print("predict...")
    submission = predict(
        pivot_df_value_log,
        pivot_value_smooth_log,
        pairs,
        Model
        )
    print("model predict complete!")

    rmse, mae, nmae = evaluate_train(df_train, Model)
    print(f"train 정확도\n rmse: {rmse}\n mae: {mae}\n nmae: {nmae}")

    answer = baseline(submission)
    if answer == "m":
        print("baseline_submission 생성완료 (Dacon/baselinez)")
    elif answer == "w":
        print("baseline_submission 생성완료 (Dacon/baseline)")
    elif answer == 1:
        print("baseline_submission 생성실패")


if __name__ == "__main__":
    print("=======main 시작=======🤞\n")
    main()
