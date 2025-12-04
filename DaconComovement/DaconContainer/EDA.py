#EDA.py
from cProfile import label

import pandas as pd
import seaborn as sns
from  data import data_load, data_preparing
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt




def EDA_run(train):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    def visualize_pair_noise(
            pivot_value,
            pivot_weight,
            pivot_value_smooth,
            pivot_weight_smooth,
            leader_id,
            follower_id,
            lag
    ):
        """
        find_comovement_pairs 함수 내부에서 계산하는 모든 파생 변수를 시각화하여
        노이즈 여부를 판별하는 함수
        """

        # 1. Leader 데이터 추출 및 가공 (사용자 코드 로직 그대로 복사)
        x_v = pivot_value.loc[leader_id].values.astype(float)
        x_w = pivot_weight.loc[leader_id].values.astype(float)
        x_sv = pivot_value_smooth.loc[leader_id].values.astype(float)
        x_sw = pivot_weight_smooth.loc[leader_id].values.astype(float)

        # Unit Price
        with np.errstate(divide='ignore', invalid='ignore'):
            x_up = np.divide(x_v, x_w, out=np.zeros_like(x_v), where=(x_w != 0))

        # Log-Diff
        x_dv = np.concatenate(([0], np.diff(np.log1p(x_v))))  # 그래프 길이를 맞추기 위해 0 padding
        x_dw = np.concatenate(([0], np.diff(np.log1p(x_w))))

        # 2. Follower 데이터 추출 (Lag 적용 안 한 원본 상태로 비교)
        y_v = pivot_value.loc[follower_id].values.astype(float)
        y_w = pivot_weight.loc[follower_id].values.astype(float)

        # Unit Price
        with np.errstate(divide='ignore', invalid='ignore'):
            y_up = np.divide(y_v, y_w, out=np.zeros_like(y_v), where=(y_w != 0))

        y_dv = np.concatenate(([0], np.diff(np.log1p(y_v))))

        # 3. 시각화 (4행 1열)
        fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
        months = np.arange(len(x_v))

        # [Plot 1] Value (우리가 맞춰야 할 정답) - 깨끗한지 확인
        ax1 = axes[0]
        ax1.plot(months, x_v, label=f'Leader Value ({leader_id})', color='blue', linewidth=2)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(months, y_v, label=f'Follower Value ({follower_id})', color='navy', linestyle='--', alpha=0.5)
        ax1.set_title(f"[Baseline] Value (Lag: {lag}) - 정답 데이터의 패턴", fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # [Plot 2] Weight (중량) - 끊기거나 튀는지 확인
        ax2 = axes[1]
        ax2.plot(months, x_w, label=f'Leader Weight', color='green')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(months, y_w, label=f'Follower Weight', color='darkgreen', linestyle='--', alpha=0.5)
        ax2.set_title("[Source: ww] Weight - 중량 데이터 품질 확인", fontsize=12)
        ax2.legend(loc='upper left')

        # [Plot 3] Unit Price (단가) - **가장 중요한 노이즈 체크 포인트**
        ax3 = axes[2]
        ax3.plot(months, x_up, label=f'Leader Unit Price (v/w)', color='red')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(months, y_up, label=f'Follower Unit Price', color='darkred', linestyle='--', alpha=0.5)
        ax3.set_title("[Source: up] Unit Price - 단가 스파이크(Noise) 확인", fontsize=12, color='red')
        ax3.legend(loc='upper left')

        # [Plot 4] Log-Diff (변화율) - 너무 랜덤한지 확인
        ax4 = axes[3]
        ax4.plot(months, x_dv, label=f'Leader Log-Diff', color='purple')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(months, y_dv, label=f'Follower Log-Diff', color='indigo', linestyle='--', alpha=0.5)
        ax4.set_title("[Source: dv] Log-Diff - 변화율의 랜덤성 확인", fontsize=12)
        ax4.legend(loc='upper left')
        ax4.axhline(0, color='black', linewidth=1)

        plt.tight_layout()
        plt.show()

    # ========================================================
    # 실행 방법
    # pairs 데이터프레임에서 상관계수가 높은 상위 1개 혹은 랜덤한 쌍을 뽑아서 확인
    # ========================================================

    # 예시: 가장 상관계수(max_corr)가 높은 쌍 뽑기
    if not pairs.empty:
        top_pair = pairs.iloc[0]  # 1등 쌍
        # top_pair = pairs.sample(1).iloc[0] # 랜덤 쌍 (여러 번 실행해보세요)

        print(f"Checking Pair: Leader({top_pair.leading_item_id}) -> Follower({top_pair.following_item_id})")
        print(f"Best Lag: {top_pair.best_lag}, Max Corr: {top_pair.max_corr}, Source: {top_pair.source}")

        visualize_pair_noise(
            pivot_df_value,
            pivot_df_weight,
            pivot_value_smooth,
            pivot_weight_smooth,
            top_pair.leading_item_id,
            top_pair.following_item_id,
            int(top_pair.best_lag)
        )
    else:
        print("No pairs found.")




    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

#value값에는 이상치가 있는것이 확실함. 하지만 없앨수는 없기때문에 정규화를 진행해야함. log1p 적용 후 standardscaler 적용해서 값을 낮추고 표준정규분포를 만들기
