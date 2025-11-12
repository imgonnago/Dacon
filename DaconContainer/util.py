from data import data_load
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocessing(train):
    train = data_load()
    scaler = StandardScaler()



