import os
import pandas as pd
import pickle


model_ml_glaucoma_dos = os.path.join(os.path.dirname(__file__), 'glaucoma_model_dos.pkl')

# Cargar el modelo 2
with open(model_ml_glaucoma_dos, 'rb') as f:
    model_training_dos = pickle.load(f)

# Lista de 5 evaluaciones con su diagnóstico final
# Formato: [Age, Sex, IOP, FamilyHistory, CataractStatus, Hypertension, Diabetes, DiagnosticoReal]
evaluaciones = [
    [53, 0, 0, 0, 0, 0, 1, 0],
    [78, 0, 1, 0, 1, 0, 1, 1],
    [74, 1, 1, 1, 1, 1, 0, 1],
    [72, 1, 1, 1, 0, 1, 0, 1],
    [66, 0, 0, 0, 1, 1, 0, 0],
    [75, 1, 0, 0, 0, 0, 0, 0],
    [73, 1, 1, 1, 0, 1, 0, 1],
    [80, 1, 1, 1, 0, 1, 0, 1],
    [49, 1, 0, 0, 0, 0, 0, 0],
    [66, 0, 0, 0, 0, 0, 0, 0],
    [69, 1, 1, 0, 1, 1, 0, 1],
    [86, 0, 1, 0, 0, 1, 0, 1],
    [79, 0, 1, 0, 1, 1, 0, 1],
    [81, 1, 1, 0, 0, 1, 0, 1],
    [48, 0, 0, 0, 0, 0, 0, 0],
    [72, 1, 1, 0, 1, 0, 0, 1],
    [62, 1, 0, 0, 0, 0, 1, 0],
    [47, 1, 1, 0, 0, 0, 0, 0],
    [70, 1, 0, 0, 1, 0, 1, 1],
    [82, 0, 1, 1, 0, 1, 0, 1],
    [52, 1, 1, 1, 0, 0, 0, 1],
    [54, 1, 1, 1, 1, 1, 0, 1],
    [65, 0, 0, 0, 0, 0, 1, 0],
    [60, 1, 1, 1, 1, 0, 0, 1],
    [71, 1, 0, 0, 0, 1, 0, 0],
    [61, 1, 1, 0, 1, 0, 1, 1],
    [85, 0, 1, 0, 0, 0, 0, 1],
    [74, 1, 1, 0, 0, 0, 1, 1],
    [65, 0, 1, 1, 0, 1, 0, 1],
    [69, 1, 1, 1, 0, 1, 0, 1]
]

# Separar datos y etiquetas reales
datos = [fila[:-1] for fila in evaluaciones]
diagnosticos_reales = [fila[-1] for fila in evaluaciones]

# Crear DataFrame para el modelo
params = pd.DataFrame(datos, columns=["Age", "Sex", "IOP", "FamilyHistory", "CataractStatus", "Hypertension", "Diabetes"])

# Hacer predicciones con modelo 2
predicciones_modelo_2 = model_training_dos.predict(params)

aciertos_modelo_2= 0

# Mostrar resultados modelo 2
for i, (row_dict, pred, real) in enumerate(zip(params.to_dict(orient="records"), predicciones_modelo_2, diagnosticos_reales), 1):
    acierto = pred == real
    print(f"Evaluación {i}: {row_dict}")
    print(f"  Diagnóstico real: {'Glaucoma' if real == 1 else 'No glaucoma'}")
    print(f"  Predicción del modelo: {'Glaucoma' if pred == 1 else 'No glaucoma'}")
    print(f"  ¿Acertó? {'Sí' if acierto else 'No'}\n")
    if acierto:
        aciertos_modelo_2 += 1

# Mostrar resumen final modelo 2
print("RESUMEN MODELO 2")
print(f"Total de evaluaciones: {len(evaluaciones)}")
print(f"Total de aciertos: {aciertos_modelo_2}")
print(f"Precisión del modelo en esta muestra: {aciertos_modelo_2 / len(evaluaciones):.2%}")