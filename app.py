import pickle
import os
import pandas as pd
import time
import pytz
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

tz = pytz.timezone('America/Lima')


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


    # prediction_time = round((end_time - start_time) * 1000, 2)

    start_time_dt = datetime.fromtimestamp(start_time, tz)
    end_time_dt = datetime.fromtimestamp(end_time, tz)

    start_time_str = start_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_time_str = end_time_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    fmt = '%Y-%m-%d %H:%M:%S.%f'

    start_dt = datetime.strptime(start_time_str, fmt)
    end_dt = datetime.strptime(end_time_str, fmt)

    diff_ms_from_str = round((end_dt - start_dt).total_seconds() * 1000, 2)


    print(end_time - start_time)


    if(result_evaluation_model[0] == 0):
        mensaje = "bajo riesgo"
    else:
        mensaje = "alto riesgo"

    print("resultado del modelo: ", result_evaluation_model[0])
    print("Tiempo de predicción:", diff_ms_from_str, "ms")
    return jsonify({
            'status': 200,
            'result_evaluation':str(result_evaluation_model[0]), 
            "message": mensaje,
            'prediction_time_ms': diff_ms_from_str,
            'start_time': start_time_str,
            'end_time': end_time_str
            })


if __name__=='__main__':
    app.run(
        host="0.0.0.0", 
        debug=True,
        port=4000 
    ) 