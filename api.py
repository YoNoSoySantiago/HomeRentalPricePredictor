from unicodedata import category
from flask import Flask, request, jsonify, render_template

import pandas as pd
import pickle

import sys
import json



app = Flask(__name__,template_folder='templates',static_folder='static')
model = pickle.load(open('final_project_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    json_data = {}
    for key in request.form:
        json_data[key] = request.form[key]
    print(json_data, file=sys.stderr)
    json_data = json_normalize(json_data)
    X = pd.DataFrame(json_data, index=[0])

    output = model.predict(X)[0]*1000
    
    return render_template('home.html',prediction_price='$ {}'.format(round(output,2)))
    #return str(round(output,2))

def json_normalize(json_data):
    """
    Helper function to flatten json
    """
    result = {}
    file = open('to_normalize.json','r')
    to_normalize = json.load(file)
    for key in json_data:
        if key in to_normalize:
            json_data[key] = int(json_data[key])
            max = to_normalize[key]['max']
            min = to_normalize[key]['min']
            result[key] = (json_data[key] - min)/(max - min)
        else:
            result[key] = json_data[key]
            
            
    return result
if __name__=="__main__":
    app.run(debug=True, port=5000)
    