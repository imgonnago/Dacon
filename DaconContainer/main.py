#main.py
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
    pairs = find_comovement_pairs(pivot_df_value)

    answer = input("EDA를 진행할까요? (y/n) >>")
    if answer == "y":
        EDA_run(train)
    elif answer == "n":
        print("EDA를 건너뜀\n")

    print("탐색된 공행성쌍 수:", len(pairs))
    print("-------pairs-------")
    print(pairs.head())
    print("\n")

    df_train = build_training_data(pivot_df_value, pivot_df_weight, pairs)
    print(df_train)
    print("=======train_x,y split complete=======\n")
    hard_voting_model = automl(df_train)
    #hard_voting_model = model()
    #fit(hard_voting_model)
    print("=======voting model fit complete=======\n")
    submission = predict(pivot_df_value,pivot_df_weight, pairs, hard_voting_model)
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
