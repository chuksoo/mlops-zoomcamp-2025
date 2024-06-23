import os
import pickle

import mlflow
from mlflow.tracking import MlflowClient

from flask import Flask, request, jsonify


MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
RUN_ID = '91c66db0c4614394ba3e03b299d190c8'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

directory_path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer')
print(f'Downloading the dict vectorizer to {directory_path}')

file_path = os.path.join(directory_path, 'dict_vectorizer.bin')
with open(file_path, 'rb') as f_out:
    dv = pickle.load(f_out)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
