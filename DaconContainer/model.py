#model.py
import xgboost as xgb
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
import lightgbm as lgbm



# 회귀모델 학습
def model():
    """extra_model = ExtraTreesRegressor(
        n_jobs=-1,
        n_estimators=47,
        max_features=1.0,
        max_leaf_nodes = 26582,
        random_state = 42
    )"""


    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        device="cpu",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=300,
        learning_rate=0.999,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        max_depth=8,
        min_child_weight=9,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'
    )


    """lgbm_model = lgbm.LGBMRegressor(
        device="gpu",
        objective='mae',
        n_jobs=-1,
        n_estimators=3412,
        num_leaves=906,
        min_child_samples=9,
        learning_rate=0.08604207947304522,
        log_max_bin=10,
        colsample_bytree=0.7899999646857239,
        reg_alpha=0.10750752847670646,
        reg_lambda=3.858904452176105,
        random_state=42
    )



    estimators = [
        ('ext', extra_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ]
    hard_voting_model = VotingRegressor(

        estimators=estimators,
        n_jobs=-1
    )"""
    return xgb_model

