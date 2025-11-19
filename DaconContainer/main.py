#main.py

from operator import concat
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
    monthly,pivot_df_value, pivot_df_weight, pivot_weight_smooth, pivot_value_smooth = data_preparing(train)
    pairs_value = find_comovement_pairs(pivot_df_value,pivot_df_value)
    pairs_weight = find_comovement_pairs(pivot_df_weight,pivot_df_value)

    pairs_value_smooth = find_comovement_pairs(pivot_value_smooth, pivot_value_smooth)
    pairs_weight_smooth = find_comovement_pairs(pivot_weight_smooth, pivot_value_smooth)

    all_pairs = pd.concat([
        pairs_value,
        pairs_weight,
        pairs_value_smooth,
        pairs_weight_smooth
    ]).drop_duplicates(subset=['leading_item_id', 'following_item_id'])
    print(all_pairs)
    print(f"Total pairs found: {len(all_pairs)}")

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
    print(all_pairs.head())
    print("\n")

    df_train = build_training_data(

        pivot_df_value,
        pivot_df_weight,
        pivot_value_smooth,
        pivot_weight_smooth,
        all_pairs
    )

    df_train = tranfrom_log_minmax(df_train)
    print(df_train)
    print("=======train_x,y split complete=======\n")
    #hard_voting_model = automl(df_train)
    hard_voting_model = model()
    te_maps = fit(hard_voting_model,df_train)
    print("=======voting model fit complete=======\n")
    submission = predict(
        pivot_df_value,
        pivot_df_weight,
        all_pairs,
        pivot_value_smooth,
        pivot_weight_smooth,
        hard_voting_model,
        te_maps)
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
    main()
