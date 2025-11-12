from  data import data_load, data_preparing
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


train = data_load()
monthly, pivot_df = data_preparing(train)

def EDA_run():
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

    # 년월에 따른 value값
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].set_title("value ")
    axes[0].scatter(monthly["ym"], monthly["value"], c='red', label='value')
    axes[0].set_xlabel('yyyy-mm')
    axes[0].set_ylabel('value')
    axes[0].legend()

    # value값의 이상치 확인
    axes[1].set_title("value ")
    axes[1].boxplot(monthly["value"], label='value')
    axes[1].set_ylabel('value')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

#value값에는 이상치가 있는것이 확실함. 하지만 없앨수는 없기때문에 정규화를 진행해야함. log1p 적용 후 standardscaler 적용해서 값을 낮추고 표준정규분포를 만들기
