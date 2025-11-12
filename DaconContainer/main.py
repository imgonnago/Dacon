from data import data_load, data_preparing, find_comovement_pairs
train = data_load()
pivot_df = data_preparing(train)
pairs = find_comovement_pairs(pivot_df)
print("탐색된 공행성쌍 수:", len(pairs))
print(pairs.head())