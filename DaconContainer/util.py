#util.py
from data import data_load
from sklearn.preprocessing import StandardScaler
from train import predict


def preprocessing(train):
    train = data_load()
    scaler = StandardScaler()

def baseline(submission):
    submission.to_csv('C:/Users/zxfg0/Dacon/baseline/baseline_submit.csv', index=False)

