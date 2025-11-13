#model.py
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# 회귀모델 학습
def model():
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=300,
        learning_rate=0.999,
        gamma=0.2,
        max_depth=8,
        min_child_weight=12,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'
    )
    return xgb_model



    #estimators = [
        #('rad', rand_model),
        #('xgb', xgb_model),
    #]
    #hard_voting_model = VotingRegressor(
        #
        #estimators=estimators
    #)
    #return hard_voting_model

