#EDA.py
from cProfile import label

import numpy as np

from  data import data_load, data_preparing
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt




def EDA_run(train):
    monthly, pivot_df_value, pivot_df_weight, pivot_weight_smooth, pivot_value_smooth = data_preparing(train)
    print("=======EDA=======\n")
    print("-------TRAIN-------")
    print(train.head())
    print("-------INFO-------")
    print(train.info())
    print("-------DESCRIBE-------")
    print(train.describe())
    print("-------CORR--------")
    print(train.drop(["item_id"],axis=1).corr())
    print("-------MONTHLY-------")
    print("-------EDA-------")
    print(monthly.head())
    print("-------INFO-------")
    print(monthly.info())
    print("-------DESCRIBE-------")
    print(monthly.describe())
    print("\n")

    pivot_df_w = np.array(pivot_df_weight)
    # 년월에 따른 value값
    plt.subplot(2,2,1)
    plt.title("month-value")
    plt.scatter(monthly['ym'],monthly['value'], color='red', label='value')
    plt.xlabel('yyyy-mm')
    plt.ylabel('value')
    plt.legend()
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.title('value-boxplot')
    plt.boxplot(monthly['value'], label='value')
    plt.ylabel('value')
    plt.legend()
    plt.grid(True)

    """plt.subplot(2,2,3)
    plt.title("ym-value-weight")
    plt.scatter(pivot_df_weight.index['ym'], , color = 'pink' ,label= 'weight')
    plt.scatter(pivot_df_value.index['ym'], pivot_df_value['value'], color = 'purple', label = 'value')
    plt.ylabel('value-weight')
    plt.xlabel('ym')
    plt.legend()
    plt.grid()
    plt.tight_layout()"""




    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

#value값에는 이상치가 있는것이 확실함. 하지만 없앨수는 없기때문에 정규화를 진행해야함. log1p 적용 후 standardscaler 적용해서 값을 낮추고 표준정규분포를 만들기
