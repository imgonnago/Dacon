from data import data_load, data_preparing, find_comovement_pairs, build_training_data
from EDA import EDA_run
from model import model
from train import predict, create_train, fit
from util import baseline


def main():
    train = data_load()
    print("=====data preparing=====")
    monthly,pivot_df = data_preparing(train)
    pairs = find_comovement_pairs(pivot_df)

    answer = input("EDA를 진행할까요? (y/n) >>")
    if answer == "y":
        EDA_run()
    elif answer == "n":
        print("EDA를 건너뜀")

    print("탐색된 공행성쌍 수:", len(pairs))
    print(pairs.head())

    build_training_data(pivot_df, pairs)
    print("train_x,y split complete")
    hard_voting_model = fit(model())
    print("voting model fit complete")
    submission = predict(pivot_df, pairs, hard_voting_model)
    print("predict complete")
    submission.head()

    baseline()
    print("baseline_submission 생성완료 (Dacon/baseline)")



if __name__ == "__main__":
    print("main 시작")
    main()
