# %%
import pandas as pd
from scipy.io import arff

# Load the ARFF file
data, meta = arff.loadarff('phpGUrE90.arff')

# Convert to pandas DataFrame
df = pd.DataFrame(data)

df.head()

# %% [markdown]
# # Dataset QSAR Biodegradation

# %%
df.info()
print("*" * 40 )
df. describe()

# %%
#Contar los nulos
nulos = df.isnull().sum()
print("Nulos por columna:\n", nulos)
##contra duplicados
duplicados = df.duplicated().sum()
print("Número de filas duplicadas:", duplicados)

# %%
#Eliminar duplicados
df = df.drop_duplicates()
print("Número de filas después de eliminar duplicados:", df.shape[0])


# %%
#Escala los valores numéricos con standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=['float64']).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df.head()

# %%
# Separamos x e y
df_temp = df.copy()
X_raw = df_temp.drop('Class', axis=1)
y_raw = df_temp['Class']

print(f"\nDatos raw X shape: {X_raw.shape}, y shape: {y_raw.shape}  ")

# %%
#codificar variable objetivo
from sklearn.preprocessing import LabelEncoder
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_raw)
print(f"Clase objetivo codificada: {le_target.classes_}")
print(f"Primeras 5 etiquetas codificadas: {y_encoded[:500]}")


# %%
# Dividir entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")



# %%


# %% [markdown]
# # Creacion del experimento en MLFlow

# %%
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tracking regresion logistica & Red Neuronal")
print("Tracking URI:", mlflow.get_tracking_uri())

# %% [markdown]
# ## Regresión Logística (Scikit-learn)

# %%
# Regresión logística probando diferentes parámetros (C, max_iter, solver)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from mlflow.models import infer_signature
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

experiment_info = {
    "objetivo": "Optimizar hiperparámetros de regresión logística para QSAR Biodegradation",
    "dataset": "QSAR Biodegradation",
    "train_size": len(X_train),
    "test_size": len(X_test),
    "features": X_train.shape[1]
}

param_grid = {
    'C': [1, 50],
    'max_iter': [50, 100],
    'solver': ['liblinear', 'lbfgs']
}

# Crear un run padre para la búsqueda de hiperparámetros; los runs de cada combinación serán sub-runs (nested=True)
with mlflow.start_run(run_name="LR_parent_search"):
    mlflow.log_param('search_type', 'grid')
    mlflow.log_param('n_C', len(param_grid['C']))
    mlflow.log_param('n_max_iter', len(param_grid['max_iter']))
    mlflow.log_param('n_solvers', len(param_grid['solver']))
    
    for C in param_grid['C']:
        for max_iter in param_grid['max_iter']:
            for solver in param_grid['solver']:
                # Cada combinación se registra como sub-run anidado
                with mlflow.start_run(run_name=f"LR_C{C}_iter{max_iter}_solver{solver}", nested=True):
                    # Crear y entrenar el modelo

                    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)
                    model.fit(X_train, y_train)

                    # Hacer predicciones
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]

                    # Calcular métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_proba)

                    # Registrar parámetros y métricas en MLflow
                    mlflow.log_param("C", C)
                    mlflow.log_param("max_iter", max_iter)
                    mlflow.log_param("solver", solver)

                    mlflow.log_metric("accuracy", accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("auc", auc)

                    # Inferir signature y agregar input_example
                    signature = infer_signature(X_test, y_pred)
                    input_example = X_test[:5]

                    mlflow.sklearn.log_model(
                        model,
                        "modelo_regresion_logistica",
                        signature=signature,
                        input_example=input_example
                    )

                    mlflow.set_tag("model_type", "LogisticRegression")
                    mlflow.set_tag("dataset", "QSAR Biodegradation")

                    mlflow.log_dict(experiment_info, "experiment_info.json")

                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Not Biodegradable', 'Biodegradable'],
                               yticklabels=['Not Biodegradable', 'Biodegradable'])
                    plt.title(f'Confusion Matrix\nC={C}, max_iter={max_iter}, solver={solver}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    confusion_path = "confusion_matrix.png"
                    plt.savefig(confusion_path)
                    plt.close()
                    mlflow.log_artifact(confusion_path)

                    print(f"Run: C={C}, max_iter={max_iter}, solver={solver} => Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

# %%


# %% [markdown]
# ## Red Neuronal (TensorFlow/Keras)

# %%
# Crea una red neuronal con Keras (modelo secuencial)
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


experiment_info_nn = {
    "objetivo": "Optimizar hiperparámetros de red neuronal para QSAR Biodegradation",
    "dataset": "QSAR Biodegradation", 
    "train_size": len(X_train),
    "test_size": len(X_test),
    "features": X_train.shape[1]
}

#Definir las capas, activaciones, optimizador y función de pérdida

mlflow.tensorflow.autolog(
    log_models=True,           # Guardar el modelo
    log_datasets=True,         # Registrar información del dataset
    log_input_examples=True,   # Guardar ejemplos de entrada
    disable=False,             # Activar autologging
    exclusive=False,           # Permitir logging manual adicional
    disable_for_unsupported_versions=False,
    silent=False
)

print("MLflow Autologging activado para TensorFlow/Keras\n")

#  INICIAR RUN DE MLFLOW
with mlflow.start_run(run_name="keras_neural_network", nested=True):
    
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
) 
    print(model.summary())

    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    #  ENTRENAR EL MODELO
    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluar el modelo - desempaquetar todas las métricas
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_precision = test_results[2]
    test_recall = test_results[3]
    test_auc = test_results[4]

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Log de métricas en MLflow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_auc", test_auc)

    #  TAGS
    mlflow.set_tags({
        "framework": "TensorFlow/Keras",
        "model_type": "Sequential Neural Network",
        "dataset": "QSAR Biodegradation"
    })

    #  LOG_DICT
    mlflow.log_dict(experiment_info_nn, "experiment_info.json")

    #  LOG_ARTIFACT - Gráfico
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Neural Network')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix_nn.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix_nn.png")

    #  LOG_TEXT
    resumen = f"""
RESULTADOS RED NEURONAL
========================
Arquitectura: 32 → 26 → 1
Activaciones: ReLU → ReLU → Sigmoid
Optimizer: Adam (lr=0.001)
Loss: Binary Crossentropy

Test Loss: {test_loss:.4f}
Test Accuracy: {test_accuracy:.4f}
Epochs: {len(history.history['loss'])}
"""
    mlflow.log_text(resumen, "resumen_nn.txt")

    print("\n Entrenamiento completo. Ejecuta 'mlflow ui' para ver resultados.")

    # Calcular y loguear f1 score

    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.4f}")

    mlflow.log_metric("f1_score", f1)

# %% [markdown]
# ## Interpretación con Ollama

# %%


mlflow.set_experiment("ollama_local")

with mlflow.start_run():
    mlflow.log_artifact("respuestas_ollama.txt")

print("✅ Archivo registrado en MLflow")

# %%


# %%



