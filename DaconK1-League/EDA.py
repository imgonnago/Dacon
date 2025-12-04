import pandas as pd

url = "https://raw.githubusercontent.com/imgonnago/Dacon/refs/heads/main/DaconK1-League/open_track1/train.csv"

data = pd.read_csv(url)
print((data.isnull().sum()/len(data)) * 100)
print()