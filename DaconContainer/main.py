from data import data_load, data_preparing, find_comovement_pairs
from EDA import EDA_run
def main():
    train = data_load()
    monthly_val,monthly_weight, pivot_df = data_preparing(train)
    EDA_run()
    pairs = find_comovement_pairs(pivot_df)
    print("탐색된 공행성쌍 수:", len(pairs))
    print(pairs.head())



if __name__ == "__main__":
    main()