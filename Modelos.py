from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calcular_especificidad(y_real, y_predicha):
    tn, fp, fn, tp = confusion_matrix(y_real, y_predicha).ravel()
    if (tn + fp) == 0: return 0
    return tn / (tn + fp)

def ejecutar_experimento(nombre_modelo, modelo_base, grilla_parametros, X_entrenamiento, y_entrenamiento, X_prueba, y_prueba, escenario):

    #X_entrenamiento: El 80% de los datos para entrenamiento y validacion
    #X_prueba:        El 20% de los datos solo para evaluacion final
    mejores_params = "Por defecto"

    if escenario == 'Sin Optimizar':
        clasificador = modelo_base
        #80%
        clasificador.fit(X_entrenamiento, y_entrenamiento)

    elif escenario == 'AG sin VC':
        #del restante se usa el 20% para validacion y 80% para entenamiento
        cv_interna = ShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

        clasificador = GASearchCV(
            estimator=modelo_base,
            cv=cv_interna,
            scoring='accuracy',
            population_size=25, generations=10,
            param_grid=grilla_parametros, n_jobs=-1, verbose=False
        )
        clasificador.fit(X_entrenamiento, y_entrenamiento)
        mejores_params = clasificador.best_params_
        clasificador = clasificador.best_estimator_

    elif escenario == 'AG con VC':
        #rotacion de los datos dentro del 80% de entrenamiento
        cv_interna = 5

        clasificador = GASearchCV(
            estimator=modelo_base,
            cv=cv_interna,
            scoring='accuracy',
            population_size=25, generations=10,
            param_grid=grilla_parametros, n_jobs=-1, verbose=False
        )
        clasificador.fit(X_entrenamiento, y_entrenamiento)
        mejores_params = clasificador.best_params_
        clasificador = clasificador.best_estimator_

    #evaluacion con el 20% guardado
    y_predicha = clasificador.predict(X_prueba)

    #ROC
    try:
        if hasattr(clasificador, "predict_proba"):
            y_probabilidad = clasificador.predict_proba(X_prueba)[:, 1]
            roc = roc_auc_score(y_prueba, y_probabilidad)
        else: roc = 0.5
    except: roc = 0.5

    acc = accuracy_score(y_prueba, y_predicha)
    sens = recall_score(y_prueba, y_predicha)
    espec = calcular_especificidad(y_prueba, y_predicha)

    tn, fp, fn, tp = confusion_matrix(y_prueba, y_predicha).ravel()
    vpp = tp / (tp + fp) if (tp + fp) > 0 else 0
    vpn = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        'Modelo': nombre_modelo,
        'Escenario': escenario,
        'Accuracy': acc,
        'Sensibilidad': sens,
        'Especificidad': espec,
        'AUC': roc,
        'VPP': vpp,
        'VPN': vpn,
        'Mejores_Parametros': mejores_params
    }


#trabajo en base al 80%
from scipy.io import arff
from sklearn.model_selection import train_test_split

try:
    data, meta = arff.loadarff('messidor_features.arff')
    df = pd.DataFrame(data)
except:
    print("¡Error! Sube el archivo messidor_features.arff al colab.")

#nombrado de columnas y correcciones
nombres_columnas = [
    'calidad', 'pre_screening', 'ma_0.5', 'ma_0.6', 'ma_0.7', 'ma_0.8', 'ma_0.9', 'ma_1.0',
    'exudados_0.5', 'exudados_0.6', 'exudados_0.7', 'exudados_0.8', 'exudados_0.9', 'exudados_1.0',
    'exudados_1.1', 'exudados_1.2', 'dist_macula_disco', 'diametro_disco', 'am_fm_class', 'clase'
]
df.columns = nombres_columnas
if df['clase'].dtype == object:
    df['clase'] = df['clase'].astype(str).str.replace("b'", "").str.replace("'", "").astype(int)

# 3. Separar X e y
X = df.drop('clase', axis=1)
y = df['clase']

#division 80% Entrenamiento, 20% Prueba Final
#X_entr_crudo 80% para entrenar y validar internamente
#X_prueba_crudo 20% intocable para la evaluación final
X_entr_crudo, X_prueba_crudo, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

#normalizacion (solo 80%)
escalador = MinMaxScaler()
escalador.fit(X_entr_crudo)

X_entr_norm = pd.DataFrame(escalador.transform(X_entr_crudo), columns=X.columns)
X_prueba_norm = pd.DataFrame(escalador.transform(X_prueba_crudo), columns=X.columns)

print(f">> Datos de Entrenamiento y Validacion (80%): {X_entr_norm.shape[0]} pacientes")
print(f">> Datos de Prueba Final (20%): {X_prueba_norm.shape[0]} pacientes")


modelo_svm = SVC(probability=True, random_state=42)
params_svm = {
    'C': Continuous(0.1, 100),
    'gamma': Categorical(['scale', 'auto']),
    'kernel': Categorical(['rbf', 'linear'])
}

modelo_mlp = MLPClassifier(max_iter=3000, random_state=42)
params_mlp = {
    'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50)]),
    'activation': Categorical(['tanh', 'relu']),
    'learning_rate_init': Continuous(0.001, 0.01)
}

modelo_rf = RandomForestClassifier(random_state=42)
params_rf = {
    'n_estimators': Integer(50, 250),
    'max_depth': Integer(5, 30)
}

Lista_Modelos = [
    ("SVM", modelo_svm, params_svm),
    ("Red Neuronal", modelo_mlp, params_mlp),
    ("Random Forest", modelo_rf, params_rf)
]

Resultados_Totales = []

variantes = [
    ("Crudos", "Sin Optimizar", X_entr_crudo, X_prueba_crudo),
    ("Normalizados", "Sin Optimizar", X_entr_norm, X_prueba_norm),
    ("Normalizados", "AG sin VC", X_entr_norm, X_prueba_norm),
    ("Normalizados", "AG con VC", X_entr_norm, X_prueba_norm)
]

print(f"{'ALGORITMO':<15} | {'CONFIGURACION':<30} | {'ACCURACY':<10} | {'SENS':<10} | {'ESPEC':<10}")
print("-" * 90)

for nombre, modelo, params in Lista_Modelos:
    for tipo_dato, escenario, X_tr, X_te in variantes:

        if tipo_dato == "Crudos": etiqueta = "1. Crudos (Sin Norm)"
        elif escenario == "Sin Optimizar": etiqueta = "2. Normalizado (Base)"
        elif escenario == "AG sin VC": etiqueta = "3. Norm + AG (Simple 80/20)"
        else: etiqueta = "4. Norm + AG (Cruzada)"

        try:
            res = ejecutar_experimento(nombre, modelo, params, X_tr, y_entrenamiento, X_te, y_prueba, escenario)

            res['Algoritmo'] = nombre
            res['Configuracion'] = etiqueta
            Resultados_Totales.append(res)

            # Imprimir resultados
            print(f"{nombre:<15} | {etiqueta:<30} | {res['Accuracy']:.4f}     | {res['Sensibilidad']:.4f}     | {res['Especificidad']:.4f}")

            # Si hay parámetros optimizados, los mostramos abajo
            if res['Mejores_Parametros'] != "Por defecto":
                print(f"   >>> Mejores Params: {res['Mejores_Parametros']}")
                print("-" * 90)

        except Exception as e:
            print(f"{nombre} - Error: {e}")


#Visuales
df_resultados = pd.DataFrame(Resultados_Totales)
df_resultados = df_resultados.sort_values(by=['Algoritmo', 'Configuracion'])


cols = ['Algoritmo', 'Configuracion', 'Accuracy', 'Sensibilidad', 'Especificidad', 'AUC', 'VPP', 'VPN']
print("TABLA RESUMEN DE MÉTRICAS:")
display(df_resultados[cols])


print("\nTABLA DE MEJORES HIPERPARÁMETROS ENCONTRADOS:")

df_params = df_resultados[df_resultados['Mejores_Parametros'] != "N/A (Por defecto)"][['Algoritmo', 'Configuracion', 'Mejores_Parametros']]
display(df_params)


metricas_grafico = [
    ('Accuracy', 'Exactitud (Accuracy)'),
    ('Sensibilidad', 'Sensibilidad (Detectar Enfermos)'),
    ('Especificidad', 'Especificidad (Detectar Sanos)')
]

sns.set_theme(style="whitegrid")

for metrica, titulo in metricas_grafico:
    plt.figure(figsize=(12, 5))
    sns.barplot(data=df_resultados, x='Algoritmo', y=metrica, hue='Configuracion', palette='viridis')
    plt.title(titulo, fontsize=14)
    plt.ylim(0.4, 1.0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.show()