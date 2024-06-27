
import sys
import boto3
import uuid
import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# Initialize a boto3 session with specific AWS credentials
session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name=''
)

# Set the boto3 session as the default for MLflow
boto3.setup_default_session(
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name=''
)

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids(len(df))
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids

def load_model(run_id):
    logged_model = f's3://mlflow-artifacts-remote-pythondance/1/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def apply_model(input_file, run_id, output_file):
    print(f'Reading the data from {input_file}')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'Loading the model with RUN_ID={run_id}')
    model = load_model(run_id)

    print(f'Applying the model')
    y_pred = model.predict(dicts)

    print(f'Saving the result to {output_file}')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration'] 
    df_result['model_version'] = run_id
    df_result.to_parquet(output_file, index=False)
    return df_result

def run():
    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 1
    

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}_tripdata_predictions_{year:04d}-{month:02}.parquet'

    RUN_ID = sys.argv[4] # '76d5937ab42549dfbc4f86b58bc7f71d'

    apply_model(
        input_file=input_file, 
        run_id=RUN_ID, 
        output_file=output_file
        )
    
    
if __name__ == '__main__':
    run()

