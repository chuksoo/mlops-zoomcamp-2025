# Load the model
import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_C=1.0.bin'

# load the model
with open(model_file, 'rb') as f_in: # read binary file
    dv, model = pickle.load(f_in)    # load() loads the file 

app = Flask('churn')


@app.route('/predict', methods=['POST'])
# make predictions
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    
    # probability that the customer will churn
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn probability': float(y_pred),
        'churn': bool(churn)
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)