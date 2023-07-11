import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

path_data = "../data/clinical.xlsx"
path_output = "../data/clinicians.csv"
df = pd.read_excel(path_data,
                   index_col=0)
df = df.drop([44, 67, 64])
y = df["Weight"]
X = df.drop(columns=["Weight_Device", "Weight", "Weight_class"])

seed = 42

kfold = StratifiedKFold(n_splits=5, shuffle=False)

results = {
    "Method": [],
    "Fold": [],
    "MAE": [],
    "R2": [],
    "MSE": [],
    "RMSE": [],
    "MAPE": []
}
for i, (train_index, test_index) in enumerate(kfold.split(X, df["Weight_class"])):
    print(f"_______Fold: {i}______")
    y_test = df["Weight"].iloc[test_index]
    y_pred = df["Weight_Device"].iloc[test_index]

    RMSE = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
    MSE = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=True)
    MAE = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    MAPE = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    R2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(f"MAE: {MAE}")
    results["Method"].append("Clinicians")
    results["Fold"].append(i)
    results["MAE"].append(MAE)
    results["R2"].append(R2)
    results["MSE"].append(MSE)
    results["RMSE"].append(RMSE)
    results["MAPE"].append(MAPE)

df_results = pd.DataFrame.from_dict(results)
df_results.to_csv(path_output)
