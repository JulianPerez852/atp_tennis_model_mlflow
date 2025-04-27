# =============================================================================
# ATP Tennis Winner Prediction
# =============================================================================
# pip install pandas numpy scikit-learn xgboost lightgbm torch torchvision mlflow

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.pytorch

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------------------------------------------------------
# 1. Configuración de MLflow
# -----------------------------------------------------------------------------
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("ATP_Tennis_Winner_Prediction")
# Asegúrate de tener MLflow corriendo localmente: `mlflow ui` en la carpeta del script.

# -----------------------------------------------------------------------------
# 2. Carga de datos y EDA
# -----------------------------------------------------------------------------
data_path = "data/atp_tennis.csv"  # Ajusta esta ruta en tu máquina
df = pd.read_csv(data_path)

print("Shape del dataset:", df.shape)
print(df.info())
print(df.head())

# Comprobación de valores faltantes
print("Valores faltantes por columna:\n", df.isnull().sum())

# Creación de la variable objetivo binaria
df['Winner_binary'] = np.where(df['Winner'] == df['Player_1'], 1, 0)
print("Distribución de la variable 'Winner_binary':\n", df['Winner_binary'].value_counts())

# Visualización rápida de la distribución
plt.figure()
df['Winner_binary'].value_counts().plot(kind='bar', title='Distribución de la etiqueta')
plt.show()

# -----------------------------------------------------------------------------
# 3. Preprocesamiento y Feature Engineering
# -----------------------------------------------------------------------------
FEATURES = ['Surface', 'Round', 'Best of', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']
TARGET  = 'Winner_binary'

X = df[FEATURES]
y = df[TARGET]

# Dividir en train/test manteniendo proporción (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definición de transformers:
# - Numéricos: Imputación (mediana) + Escalado (StandardScaler)
# - Categóricos: Imputación (valor 'missing') + One-Hot Encoding
numeric_feats = ['Best of', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),           # Manejo de missing
    ('scaler', StandardScaler())                             # Normalización
])

categorical_feats = ['Surface', 'Round']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))       # Feature engineering
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_feats),
    ('cat', categorical_transformer, categorical_feats)
])

# -----------------------------------------------------------------------------
# 4. Experimento 1: Modelos clásicos con Regularización
#    - Logistic Regression L2 (Ridge) y L1 (Lasso)
# -----------------------------------------------------------------------------
for penalty in ['l2', 'l1']:
    with mlflow.start_run(run_name=f"LogReg_{penalty}"):
        # Pipeline completo
        model = Pipeline([
            ('preproc', preprocessor),
            ('clf', LogisticRegression(
                penalty=penalty,
                solver='saga',        # soporta L1 y L2
                max_iter=2000,
                random_state=42
            ))
        ])
        # GridSearch CV para lambda (C inverso de regularización)
        param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
        search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        
        # Predicciones
        y_pred  = best.predict(X_test)
        y_prob  = best.predict_proba(X_test)[:,1]
        
        # Métricas
        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Log params & metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("penalty", penalty)
        mlflow.log_param("best_C", search.best_params_['clf__C'])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Guardar modelo
        mlflow.sklearn.log_model(best, "model")
        print(f"[LogReg {penalty}] ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")

# -----------------------------------------------------------------------------
# 5. Experimento 2: Ensamblados
#    - Random Forest, XGBoost, LightGBM
# -----------------------------------------------------------------------------
models_ens = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost":       xgb.XGBClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42
                    ),
    "LightGBM":      lgb.LGBMClassifier(
                        n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42
                    )
}

for name, clf in models_ens.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([
            ('preproc', preprocessor),
            ('clf', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]
        
        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(pipe, "model")
        
        print(f"[{name}] ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")

# -----------------------------------------------------------------------------
# 6. Experimento 3: Red Neuronal en PyTorch
#    - MLP con BatchNorm y Dropout, Optimizador Adam, Early Stopping
# -----------------------------------------------------------------------------
# Preparar tensores
# Transformación de datos con el pipeline preprocessor
X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)

# Crear DataLoaders
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds = TensorDataset(torch.tensor(X_train_t, dtype=torch.float32),
                            torch.tensor(y_train.values, dtype=torch.float32))
test_ds  = TensorDataset(torch.tensor(X_test_t, dtype=torch.float32),
                            torch.tensor(y_test.values, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

# Definición del modelo MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),    # BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),        # Dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP(X_train_t.shape[1]).to(device)
criterion = nn.BCELoss()                     # Cross-entropy para binaria
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping setup
best_roc = 0.0
patience = 5
counter  = 0
best_model_path = "best_model.pt"

# Active MLflow autolog para PyTorch, pero sin registrar modelo automáticamente
mlflow.pytorch.autolog(log_models=False)

with mlflow.start_run(run_name="PyTorch_MLP_Optimizado"):
    # Registrar parámetros manuales relevantes
    mlflow.log_param("dropout", 0.3)
    mlflow.log_param("batch_size", 256)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("early_stopping_patience", patience)

    # Bucle de entrenamiento con early stopping
    for epoch in range(1, 51):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            preds = model(xb)
            loss  = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluación en test
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                prob = model(xb).cpu().numpy()
                all_probs.extend(prob.flatten().tolist())
                all_labels.extend(yb.numpy().tolist())
        all_preds = [1 if p > 0.5 else 0 for p in all_probs]

        # Cálculo de métricas
        roc_auc = roc_auc_score(all_labels, all_probs)
        f1      = f1_score(all_labels, all_preds)
        # Métricas serán registradas automáticamente por autolog

        print(f"Epoch {epoch:02d} | ROC AUC: {roc_auc:.4f} | F1-score: {f1:.4f}")

        # Early stopping y checkpoint del mejor modelo
        if roc_auc > best_roc:
            best_roc = roc_auc
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping: no hay mejora tras {} épocas.".format(patience))
                break

    # Al finalizar, cargar y registrar sólo una vez el mejor modelo
    model.load_state_dict(torch.load(best_model_path))
    mlflow.pytorch.log_model(model, artifact_path="model")

    print(f"Mejor ROC AUC obtenido: {best_roc:.4f}")

# -----------------------------------------------------------------------------
# 7. Comparativa de resultados (usando MLflow Tracking UI o API)
# -----------------------------------------------------------------------------
# Desde línea de comando o notebook: abrir http://localhost:5000 en tu navegador
print("Todos los experimentos han finalizado. Abre MLflow UI con `mlflow ui` y selecciona el mejor modelo según ROC AUC.")