#model.py
import xgboost as xgb
from catboost import CatBoostRegressor
from pandas.core.common import random_state
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
import lightgbm as lgbm



# 회귀모델 학습
def model(df_train):
    X = df_train[["b_t", "b_t_1", "a_t_lag", "max_corr", "best_lag"]]

    y = df_train["target"]

    """extra_model = ExtraTreesRegressor(
        n_jobs=-1,
        n_estimators=47,
        max_features=1.0,
        max_leaf_nodes = 26582,
        random_state = 42
    )"""


    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        device="gpu",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=300,
        learning_rate=0.09999,
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
        device="cpu",
        objective='mae',
        n_jobs=-1,
        n_estimators=500,
        num_leaves=906,
        min_child_samples=9,
        learning_rate=0.1,
        max_bin=10,
        colsample_bytree=0.7899999646857239,
        reg_alpha=0.10750752847670646,
        reg_lambda=3.858904452176105,
        random_state=42
    )"""

    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=7,
        loss_function='MAE',
        random_state=42,
        verbose=100,
        allow_writing_files=False,
        l2_leaf_reg=3,
        bagging_temperature=1,
        task_type='GPU',
        early_stopping_rounds=100
    )


    return cat_model, xgb_model

# 회귀모델 학습
def get_xgb_model():
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        tree_method="hist",
        n_estimators=300,
        learning_rate=0.09999,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        max_depth=8,
        min_child_weight=9,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'
    )
    return xgb_model

# 2. CatBoost 모델
def get_cat_model():
    cat_model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.12,
        depth=9,
        loss_function='MAE',
        random_state=42,
        verbose=100,
        allow_writing_files=False,
        l2_leaf_reg=3,
        bagging_temperature=1,
        early_stopping_rounds=100
    )
    return cat_model

