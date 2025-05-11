import pickle
import os
import pandas as pd
import numpy as np
import time
from flask import Flask, jsonify, request
from flask_cors import CORS


# model_ml_glaucoma = r'D:/EyeAlertModeloAPI/glaucoma_model.pkl'
model_ml_glaucoma = os.path.join(os.path.dirname(__file__), 'glaucoma_model.pkl')


with open(model_ml_glaucoma, 'rb') as file:
    model_training = pickle.load(file)


app = Flask(__name__)
CORS(app)

@app.route('/')
def ping():
    return jsonify({'message':'API model glaucoma'})



@app.route('/evaluation',methods=['POST'])
def addEvaluationUser():
    Age=request.json.get('age')
    Sex=request.json.get('sex')
    IOP=request.json.get('iop')
    FamilyHistory=request.json.get('familyHistory')
    CataractStatus=request.json.get('cataractStatus')
    Hypertension=request.json.get('hypertension')
    Diabetes=request.json.get('diabetes')


    params = pd.DataFrame([[Age, Sex, IOP, FamilyHistory, CataractStatus, Hypertension, Diabetes]], columns= ["Age", "Sex", "IOP", "FamilyHistory", "CataractStatus", "Hypertension", "Diabetes"])

    print(params)

    start_time = time.time()
    result_evaluation_model = model_training.predict(params)
    end_time = time.time()

    prediction_time = round((end_time - start_time) * 1000, 2)


    if(result_evaluation_model[0] == 0):
        mensaje = "bajo riesgo"
    else:
        mensaje = "alto riesgo"

    print("resultado del modelo: ", result_evaluation_model[0])
    print("Tiempo de predicción:", prediction_time, "ms")
    return jsonify({
            'status': 200,
            'result_evaluation':str(result_evaluation_model[0]), 
            "message": mensaje,
            'prediction_time_ms': prediction_time 
            })


if __name__=='__main__':
    app.run(
        host="0.0.0.0", 
        debug=True,
        port=4000 
    ) 