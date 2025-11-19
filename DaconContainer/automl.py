

from flaml import AutoML
import time

def automl(df_train_model):
    automl = AutoML(n_jobs=-1,gpu_per_trial= -1)

    settings = {
        "time_budget": 600,
        "task": "regression",
        "metric": "mae",
        "estimator_list":  ['lgbm', 'xgboost', 'extra_tree', 'xgb_limitdepth', 'sgd', 'catboost'],
        "log_file_name": "automl_regression.log",

    }

    automl.fit(
        dataframe=df_train_model,
        label = 'target',
        **settings
    )
    print("\n" + "=" * 50)
    print(" " * 20 + "AUTOML 요약")
    print("=" * 50)
    print("Best estimator (모델타입):", automl.best_estimator)
    print("Best loss(mse 기준):", automl.best_loss)
    print("\nBest config(하이퍼파라미터)")
    print(automl.best_config)
    print("=" * 50)
    print("5초간 대기 후 다음 단계로 넘어갑니다...")
    time.sleep(5)

    return automl