import pickle
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request


model_ml_glaucoma = r'D:/EyeAlertModeloAPI/glaucoma_model.pkl'

with open(model_ml_glaucoma, 'rb') as file:
    model_training = pickle.load(file)


app = Flask(__name__)


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

    result_evaluation_model = model_training.predict(params)


    if(result_evaluation_model[0] == 0):
        mensaje = "bajo riesgo"
    else:
        mensaje = "alto riesgo"

    print("resultado del modelo: ", result_evaluation_model[0])
    return jsonify({'result_evaluation:':str(result_evaluation_model), "message:": mensaje})


if __name__=='__main__':
    app.run(
        host="0.0.0.0", #PUERTO PARA ACCEDER DESDE FLUTTER
        debug=True,#Al realizar cambios, la app se reinicia
        port=4000 #PUERTO A USAR, POR DEFECTO ES 5000
    ) 