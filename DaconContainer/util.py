#util.py
from data import data_load
from sklearn.preprocessing import StandardScaler
from train import predict

def baseline(submission):

    submission['abs_corr'] = submission['max_corr'].abs()
    submission = submission.sort_values(by='abs_corr', ascending=False)

    final_predictions = submission.drop_duplicates(
        subset=["leading_item_id", "following_item_id"],
        keep="first"
    )
    final_predictions = final_predictions[['leading_item_id', 'following_item_id', 'value']]

    answer = input("mac/window(m/w)")
    if answer == "m":
        print("sumission.csv is saved to mac")
        final_predictions.to_csv('/Users/joyongjae/Dacon/baseline/baseline_submit.csv', index=False)
        print("complete")
        return answer

    elif answer == "w":
        print("submission.csv is saved to window")
        final_predictions.to_csv('C:/Users/zxfg0/Dacon/baseline/baseline_submit.csv', index=False)
        print("complete")
        return answer
    else:
        print("you didn't choose os enviroment")
        return 1

