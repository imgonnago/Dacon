#util.py
from data import data_load
from sklearn.preprocessing import StandardScaler
from train import predict

def baseline(submission):
    answer = input("mac/window(m/w)")
    if answer == "m":
        print("sumission.csv is saved to mac")
        submission.to_csv('/Users/joyongjae/Dacon/baseline/baseline_submit.csv', index=False)
        print("complete")
        return answer

    elif answer == "w":
        print("submission.csv is saved to window")
        submission.to_csv('C:/Users/zxfg0/Dacon/baseline/baseline_submit.csv', index=False)
        print("complete")
        return answer
    else:
        print("you didn't choose os enviroment")
        return 1

