# Load the model
import pickle

input_file = 'model_C=1.0.bin'

# load the model
with open(input_file, 'rb') as f_in: # read binary file
    dv, model = pickle.load(f_in)    # load() loads the file 

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

X = dv.transform([customer])

# probability that the customer will churn
y_pred = model.predict_proba(X)[0, 1]

print('User Input:', customer)
print('Churn Probability:', y_pred)

