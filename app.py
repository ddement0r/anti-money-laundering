from flask import Flask, jsonify, request
import pickle
from threading import Thread
import pandas as pd

app = Flask(__name__)

# Load the model from the .pkl file
with open('antilaundering.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET'])
def test():
    return "hi hiii"
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    type = data["type"]
    amount = data["amount"]
    oldbalanceOrg = data["oldbalanceOrg"]
    newbalanceOrig = data["newbalanceOrig"]
    oldbalanceDest = data["oldbalanceDest"]
    newbalanceDest = data["newbalanceDest"]



    entry = {"type": [type], "amount": [amount], "oldbalanceOrg": [oldbalanceOrg], "newbalanceOrig": [newbalanceOrig], "oldbalanceDest": [oldbalanceDest],"newbalanceDest":[newbalanceDest]}
    entry_frame = pd.DataFrame.from_dict(entry)

    predictions =  model.predict(entry_frame)[0] 
    print(predictions)
    isFraud = "Fraud" if predictions == 1 else "Not Fraud"
    try: 
        return {'predictions': isFraud}, 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run():
    app.run(port=8000)

thread = Thread(target=run)
thread.start()
