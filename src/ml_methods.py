import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

path_data = "../data/clinical.csv"
path_output = "../data/ml_methods.csv"

df = pd.read_csv(path_data, sep=",", index_col=0)

y = df["Weight"]
X = df.drop(columns=["Weight_Device", "Weight", "Weight_class"])

models = [("LR", LinearRegression), ("XGB", XGBRegressor)]

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

for name, model in models:
    print(f"_______{name}______")
    for i, (train_index, test_index) in enumerate(kfold.split(X, df["Weight_class"])):
        print(f"_______Fold: {i}______")
        x_train = X.iloc[train_index, :]
        x_test = X.iloc[test_index, :]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        clf = model()
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        RMSE = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
        MSE = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=True)
        MAE = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        MAPE = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
        R2 = r2_score(y_true=y_test, y_pred=y_pred)
        print(f"MAE: {MAE}")
        results["Method"].append(name)
        results["Fold"].append(i)
        results["MAE"].append(MAE)
        results["R2"].append(R2)
        results["MSE"].append(MSE)
        results["RMSE"].append(RMSE)
        results["MAPE"].append(MAPE)

df_results = pd.DataFrame.from_dict(results)
df_results.to_csv(path_output)
