import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Cargar tu dataset
df = pd.read_csv("dataset_clean_glaucoma_dos.csv")

# Preparar datos
X = df.drop(columns=["Unnamed: 0", "Glaucoma"])
y = df["Glaucoma"]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Entrenar modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Obtener importancia de características
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(importances)
