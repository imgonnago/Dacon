from flaml import AutoML

def automl(df_train_model):
    feature_cols = ['b_t', 'b_t_1', 'a_t_lag','a_t_lag_weight', 'max_corr', 'best_lag']
    train_X = df_train_model[feature_cols].values
    train_y = df_train_model["target"].values
    automl = AutoML()

    settings = {
        "time_budget": 120,
        "task": "regression",
        "metric": "mse",
        "estimator_list": ["xgboost"],
        "log_file_name": "automl_multi.log",
    }

    automl.fit(
        X_train=train_X,
        y_train=train_y,
        **settings
    )
    print("요약")
    print("Best estimator (모델타입):" ,automl.best_estimator)
    print("Best loss(mse 기준):" ,automl.best_loss)
    print("\nBest config(하이퍼파라미터)")
    print(automl.best_config)

    return automl