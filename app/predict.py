import os

from flask import Flask, request, render_template, jsonify
import pickle
import json

import numpy as np
import pandas as pd

# import sys
# sys.path.insert(1, os.getcwd()+'\\train\\')

# print(sys.path)
# print(os.getcwd())

from features import num_features, cat_features

app = Flask(__name__)

model_path = ""
if "app" in str(os.getcwd()) or "train" in str(os.getcwd()): 
    os.chdir("..//saved_models")
    model_path = os.getcwd()
else:
    model_path = os.getcwd()+"//saved_models"

model = pickle.load(open(model_path+"//model.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('info.html')

@app.route('/predict', methods=['GET'])
def predict():
    if not request.json:
        return jsonify({prediction:"No test data received"})
    
    r = json.dumps(request.json).replace('', "")
    input_json = json.loads(r)
    x_arr = parseArgs(input_json)

    # Predict on x_array and return JSON response.
    estimate = round(float(model.predict(x_arr)[0]),2)
    response = dict(Prediction=estimate)

    return jsonify(response)
    

def parseArgs(request_dict):
    
    x_df = pd.DataFrame(request_dict, index=[0])
    x_df = x_df.replace('', np.nan, regex=True)
    x_df[cat_features] = x_df[cat_features].astype('category')
    x_df[num_features] = x_df[num_features].astype('float64')
    # x_list = []
    # for feature in list(num_features + cat_features):
    #     value = request_dict.get(feature, None)
    #     if value != np.nan:
    #         x_list.append(value)
    #     else:
    #         # Handle missing features.
    #         x_list.append(np.nan)
    return x_df

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)