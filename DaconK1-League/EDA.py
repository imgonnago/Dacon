import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
url = "https://raw.githubusercontent.com/imgonnago/Dacon/refs/heads/main/DaconK1-League/open_track1/train.csv"

data = pd.read_csv(url)
print(data.describe())
print(data.head())
print(data.info())
#결측치 확인
print(data['result_name'])
print((data.isnull().sum()/len(data)) * 100)
print(data['result_name'].isnull().sum())
#sns.pairplot(data[['player_id','action_id','start_x','start_y','end_x','end_y']])
#plt.show()
#선수별 시작 x,y좌표
print(data.groupby(['player_id'])[['start_x','start_y']].mean().head())
"""포지션별로 나눠서 하면 예측하기 수월할 것.105*68
0~15m: goalkeeper
15~40: defender
40~60m: midfilder
60~80m: attacker"""
goalkeeper = data[(data['start_x'] <= 15) & (data['start_y'] >= 20) & (data['start_y'] <= 40)]
#right
defender_r = data[(data['start_x'] > 15) & (data['start_x'] <= 40) & (data['start_y'] <= 24)]
midfilder_r = data[(data['start_x'] >= 40) & (data['start_x'] <= 60) & (data['start_y'] <= 24)]
attacker_r = data[(data['start_x'] >= 60) & (data['start_x'] <= 80) & (data['start_y'] <= 24)]
#left
defender_l = data[(data['start_x'] > 15) & (data['start_x'] <= 40) & (data['start_y'] >= 44) & (data['start_y'] <= 68)]
midfilder_l = data[(data['start_x'] >= 40) & (data['start_x'] <= 60) & (data['start_y'] >= 44) & (data['start_y'] <= 68)]
attacker_l = data[(data['start_x'] >= 60) & (data['start_x'] <= 80) & (data['start_y'] >= 44) & (data['start_y'] <= 68)]
#sort
goalkeeper_sorted = goalkeeper.sort_values(by='player_id', ascending=False)
#right sorted
defender_r_sorted = defender_r.sort_values(by='player_id', ascending=False)
midfilder_r_sorted = midfilder_r.sort_values(by='player_id', ascending=False)
attacker_r_sorted = attacker_r.sort_values(by='player_id', ascending=False)
#left_sorted
defender_l_sorted = defender_l.sort_values(by='player_id', ascending=False)
midfilder_l_sorted = midfilder_l.sort_values(by='player_id',ascending=False)
attacker_l_sorted = attacker_l.sort_values(by='player_id', ascending=False)
print('='*20)
print("goalkeeper")
print('='*20)
print(goalkeeper_sorted[['team_id','player_id','action_id','start_x','start_y','end_x','end_y']].head())
print('='*20)
print("right defender")
print('='*20)
print(defender_r_sorted[['team_id','player_id','action_id','start_x','start_y','end_x','end_y']].head())
print('='*20)
print("right midfilder")
print('='*20)
print(midfilder_r_sorted[['team_id','player_id','action_id','start_x','start_y','end_x','end_y']].head())
print('='*20)
print("right attacker")
print('='*20)
print(attacker_r_sorted[['team_id','player_id','action_id','start_x','start_y','end_x','end_y']].head())
print('='*20)
print("left defender")
print('='*20)
print(defender_l_sorted[['team_id','player_id','action_id','start_x','start_y','end_x','end_y']].head())
print('='*20)
print("left midfilder")
print('='*20)
print(midfilder_l_sorted[['team_id','player_id','action_id','start_x','start_y','end_x','end_y']].head())
print('='*20)
print("left attacker")
print('='*20)
print(attacker_l_sorted[['team_id','player_id','action_id','start_x','start_y','end_x','end_y']].head())

team_data_sorted = data.sort_values(by=['player_id','team_id'], ascending = False)
print(team_data_sorted[['team_id','player_id','start_x','start_y','end_x','end_y']].head())