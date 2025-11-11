from cProfile import label
from itertools import groupby

import pandas as pd
import matplotlib
from pandas import pivot
from xgboost import train

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import lineStyles

train_url = "https://raw.githubusercontent.com/imgonnago/Dacon/refs/heads/main/ACD2-Week12-1/dataset/train.csv"
train = pd.read_csv(train_url)

monthly = (
    train
    .groupby(["item_id", "year", "month"], as_index=False)["value"]
    .sum()
)

# year, month를 하나의 키(ym)로 묶기
monthly["ym"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
)
monthly
print("데이터 로드 성공")
print("-------EDA-------")
print(monthly.head())
print("-------INFO-------")
print(monthly.info())
print("-------DESCRIBE-------")
print(monthly.describe())

#년월에 따른 value값
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].set_title("value data")
axes[0].scatter(monthly["ym"], monthly["value"], c='red', label = 'value')
axes[0].set_xlabel('yyyy-mm')
axes[0].set_ylabel('value')
axes[0].legend()

#value값의 이상치 확인
axes[1].set_title("value data")
axes[1].boxplot(monthly["value"], label = 'value')
axes[1].set_ylabel('value')
axes[1].legend()
plt.tight_layout()

print(train.info())
print(train.head())

#value값에는 이상치가 있는것이 확실함. 하지만 없앨수는 없기때문에 정규화를 진행해야함. log1p 적용 후 standardscaler 적용해서 값을 낮추고 표준정규분포를 만들기
#mopdified