import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred, squared=True)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, rmse, mae, mape


def get_fold_metrics(results):
    df = results.groupby(["patient_id"]).mean().reset_index()
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    mse, rmse, mae, mape = get_metrics(y_true, y_pred)
    return mse, rmse, mae, mape


def save_fold_metrics(experiment, results, fold):
    mse, rmse, mae, mape = get_fold_metrics(results)
    df_table = pd.DataFrame(
        {"fold": [fold], "mse": [mse], "rmse": [rmse], "mae": [mae], "mape": [mape]})
    experiment.log_table(f"fold_{fold}_metrics.csv", df_table)


def save_metrics_all(experiment, fold_results):
    folds = []
    mses = []
    rmses = []
    maes = []
    mapes = []
    for fold, results in fold_results.items():
        mse, rmse, mae, mape = get_fold_metrics(results)
        folds.append(fold)
        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        mapes.append(mape)

    folds.append("mean")
    mses.append(np.array(mses).mean())
    rmses.append(np.array(rmses).mean())
    maes.append(np.array(maes).mean())
    mapes.append(np.array(mapes).mean())

    folds.append("std")
    mses.append(np.array(mses).std())
    rmses.append(np.array(rmses).std())
    maes.append(np.array(maes).std())
    mapes.append(np.array(mapes).std())

    df_table = pd.DataFrame(
        {"fold": folds, "mse": mses, "rmse": rmses, "mae": maes, "mape": mapes})
    experiment.log_table(f"all_folds_metrics.csv", df_table)


def save_per_patient_results(experiment, fold_results):
    df = pd.concat((res for _, res in fold_results.items()))

    df2 = df.copy()
    df2 = df2.groupby(["patient_id"]).mean().reset_index()
    df2["body_part"] = "all"
    df = pd.concat([df, df2])
    grouped_bp = df.groupby(["patient_id", "body_part"]).mean().reset_index()
    grouped_bp["fold"] = (grouped_bp["fold"] + 1).astype(int)

    grouped_bp_all = grouped_bp.copy()
    grouped_bp_all["fold"] = "All"

    grouped_bp = pd.concat([grouped_bp, grouped_bp_all])
    experiment.log_table(f"per_patient_pred.csv", grouped_bp)
