from cProfile import label
from itertools import groupby
import pandas as pd
import matplotlib
from  data import data_load, data_preparing


#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import lineStyles

train = data_load()
monthly_data, pivot_df = data_preparing(train)

def EDA_run():
    print("데이터 로드 성공")
    print("-------TRAIN-------")
    print(train.head())
    print("-------INFO-------")
    print(train.info())
    print("-------DESCRIBE-------")
    print(train.describe())
    print("-------MONTHLY-------")
    print("-------EDA-------")
    print(monthly_data.head())
    print("-------INFO-------")
    print(monthly_data.info())
    print("-------DESCRIBE-------")
    print(monthly_data.describe())

    # 년월에 따른 value값
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].set_title("value ")
    axes[0].scatter(monthly_data["ym"], monthly_data["value"], c='red', label='value')
    axes[0].set_xlabel('yyyy-mm')
    axes[0].set_ylabel('value')
    axes[0].legend()

    # value값의 이상치 확인
    axes[1].set_title("value ")
    axes[1].boxplot(monthly_data["value"], label='value')
    axes[1].set_ylabel('value')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

#value값에는 이상치가 있는것이 확실함. 하지만 없앨수는 없기때문에 정규화를 진행해야함. log1p 적용 후 standardscaler 적용해서 값을 낮추고 표준정규분포를 만들기
