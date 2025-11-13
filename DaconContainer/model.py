#model.py
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# 회귀모델 학습
def model():
    rand_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=10,
        min_samples_split=5,
        random_state=42
    )

    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=300,
        learning_rate=0.1,
        gamma=0.1,
        max_depth=9,
        min_child_weight=12,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'
    )

    estimators = [
        ('rad', rand_model),
        ('xgb', xgb_model),
    ]
    hard_voting_model = VotingRegressor(
        estimators=estimators
    )
    return hard_voting_model

