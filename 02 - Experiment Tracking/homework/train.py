import os
import pickle
import click
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # enable autologging with mlflow
    with mlflow.start_run():
        mlflow.set_tag("developer", "chuks")
        mlflow.set_tag("model", "randomforest") 
        mlflow.set_tag("logtype", "auto") 
        mlflow.log_param("train-data-path", os.path.join(data_path, "train.pkl"))
        mlflow.log_param("valid-data-path", os.path.join(data_path, "val.pkl"))
    
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # log the model parameters used for this run
        mlflow.log_param("max_depth", rf.max_depth)
        mlflow.log_param("random_state", rf.random_state)
        mlflow.log_param("min_samples_split", rf.min_samples_split)

        # log the RMSE metrics used for this run
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Log the model created by this run.
        mlflow.sklearn.log_model(rf, "random-forest-model") 


if __name__ == '__main__':
    run_train()
