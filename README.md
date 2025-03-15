# EyeAlert API REST del Modelo Predictivo

Este es el backend del modelo predictivo para la aplicación EyeAlert para evaluar el nivel de riesgo de Glaucoma.

## ¿Cómo correr la aplicación en un entorno local?

Primero creamos un entorno virtual usando el comando

```
py -m venv venv
```

Navegamos hasta el directorio `Scripts` dentro de `venv`

```
cd .\vemv\Scripts\
```

Ejecutamos el archivo active

```
.\active
```

Una vez entramos en el entorno virtual, nos dirigimos a la raíz del proyecto `EyeAlertModeloAPI`

```
cd ..
cd ..
```

Instalamos las diferentes librerías necesarias

## (OPCIONAL) Librerías para la limpieza del dataset y entremiento del modelo 

```
pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install plotly
pip install scikit-learn
pip install xgboost
```

## Librerías para la API REST del modelo predictivo

```
pip install pandas
pip install numpy
pip install pickle
pip install flask
pip install flask_cors
```

## Levantamos el servidor 

```
python app.py
```

