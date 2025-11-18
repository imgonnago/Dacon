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
    item_to_hs4_map = train.set_index('item_id')['hs4'].to_dict()
    print(train.head())
    print("=====data preparing=====")
    monthly,pivot_df_value, pivot_df_weight = data_preparing(train)
    pairs_value = find_comovement_pairs(pivot_df_value,pivot_df_value)
    pairs_weight = find_comovement_pairs(pivot_df_weight,pivot_df_value)

    pivot_value_smooth = pivot_df_value.rolling(window=3, axis=1).mean().fillna(0)
    pivot_weight_smooth = pivot_df_weight.rolling(window=3, axis=1).mean().fillna(0)

    pairs_value_smooth = find_comovement_pairs(pivot_value_smooth, pivot_value_smooth)
    pairs_weight_smooth = find_comovement_pairs(pivot_weight_smooth, pivot_value_smooth)
    all_pairs = pd.concat([
        pairs_value,
        pairs_weight,
        pairs_value_smooth,
        pairs_weight_smooth
    ]).drop_duplicates(subset=['leading_item_id', 'following_item_id'])

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

    df_train = build_training_data(item_to_hs4_map, pivot_df_value, pivot_df_weight, all_pairs)
    df_train, x_scaler, y_scaler = tranfrom_log_minmax(df_train)
    print(df_train)
    print("=======train_x,y split complete=======\n")
    hard_voting_model = automl(df_train)
    #hard_voting_model = model()
    #fit(hard_voting_model,df_train)
    print("=======voting model fit complete=======\n")
    submission = predict(pivot_df_value,pivot_df_weight, all_pairs, hard_voting_model,x_scaler,y_scaler,item_to_hs4_map)
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
