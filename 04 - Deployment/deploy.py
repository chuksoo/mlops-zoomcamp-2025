
# customer information
customer_id = "asdx-123d"
customer_email = "asdx-123d@yahoo.com"
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "two_year",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": (2 * 29.85)
}

# Making requests
import requests
url = "http://localhost:9696/predict"
response = requests.post(url, json=customer).json()
print(response)

if response["churn"]:
    print(f"Sending email to {customer_id} with email:", {customer_email})
else:
    print(f"Customer {customer_id} will not churn")
