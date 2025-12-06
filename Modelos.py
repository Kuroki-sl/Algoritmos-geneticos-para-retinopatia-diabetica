# =============================================================================
# 1. IMPORTACIONES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Para guardar el escalador

from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score

# Algoritmos de Machine Learning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Algoritmos Genéticos
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer

# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def calcular_especificidad(y_real, y_predicha):
    tn, fp, fn, tp = confusion_matrix(y_real, y_predicha).ravel()
    if (tn + fp) == 0: return 0
    return tn / (tn + fp)

def ejecutar_experimento(nombre_modelo, modelo_base, grilla_parametros, X_train, y_train, X_test, y_test, escenario):
    """
    Función central que entrena y evalúa los modelos según el escenario.
    """
    mejores_params = "N/A (Por defecto)"

    # Clonamos el modelo base para no sobrescribir configuraciones previas
    from sklearn.base import clone
    clasificador = clone(modelo_base)

    if escenario == 'Sin Optimizar':
        # Entrenamos con el 80% directo
        clasificador.fit(X_train, y_train)

    elif escenario == 'AG sin VC':
        # Split simple interno para el AG (del 80%, saca un trozo para validar)
        cv_interna = ShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

        gas = GASearchCV(
            estimator=modelo_base,
            cv=cv_interna,
            scoring='accuracy',
            population_size=10, generations=5,  # Ajusta esto si quieres más potencia
            param_grid=grilla_parametros, n_jobs=-1, verbose=False
        )
        gas.fit(X_train, y_train)
        mejores_params = gas.best_params_
        clasificador = gas.best_estimator_

    elif escenario == 'AG con VC':
        # Validación cruzada de 5 pliegues sobre el 80%
        cv_interna = 5

        gas = GASearchCV(
            estimator=modelo_base,
            cv=cv_interna,
            scoring='accuracy',
            population_size=10, generations=5,
            param_grid=grilla_parametros, n_jobs=-1, verbose=False
        )
        gas.fit(X_train, y_train)
        mejores_params = gas.best_params_
        clasificador = gas.best_estimator_

    # --- EVALUACIÓN FINAL CON EL 20% RESERVADO ---
    y_predicha = clasificador.predict(X_test)

    # Cálculo de métricas
    try:
        if hasattr(clasificador, "predict_proba"):
            y_prob = clasificador.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_prob)
        else: roc = 0.5
    except: roc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_test, y_predicha).ravel()

    return {
        'Modelo': nombre_modelo,
        'Escenario': escenario,
        'Accuracy': accuracy_score(y_test, y_predicha),
        'Sensibilidad': recall_score(y_test, y_predicha),
        'Especificidad': calcular_especificidad(y_test, y_predicha),
        'AUC': roc,
        'VPP': tp / (tp + fp) if (tp + fp) > 0 else 0, # Precisión
        'VPN': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'Mejores_Parametros': mejores_params
    }


# =============================================================================
# 3. CARGA, LIMPIEZA, DIVISIÓN Y NORMALIZACIÓN
# =============================================================================

# A. Carga de datos
try:
    data, meta = arff.loadarff('messidor_features.arff')
    df = pd.DataFrame(data)
except:
    print("¡Error! Asegúrate de tener el archivo 'messidor_features.arff'.")
    # Generamos datos dummy para que el código no rompa si falta el archivo
    from sklearn.datasets import make_classification
    X_dum, y_dum = make_classification(n_samples=1151, n_features=19, random_state=42)
    df = pd.DataFrame(X_dum)
    df['clase'] = y_dum

# Limpieza básica de nombres de columnas
nombres_columnas = [
    'calidad', 'pre_screening', 'ma_0.5', 'ma_0.6', 'ma_0.7', 'ma_0.8', 'ma_0.9', 'ma_1.0',
    'exudados_0.5', 'exudados_0.6', 'exudados_0.7', 'exudados_0.8', 'exudados_0.9', 'exudados_1.0',
    'exudados_1.1', 'exudados_1.2', 'dist_macula_disco', 'diametro_disco', 'am_fm_class', 'clase'
]
if df.shape[1] == 20: df.columns = nombres_columnas

# Corrección de tipo de dato en la clase (de bytes a enteros si es necesario)
if df['clase'].dtype == object:
    df['clase'] = df['clase'].astype(str).str.replace("b'", "").str.replace("'", "").astype(int)

# -------------------------------------------------------------------------
# LIMPIEZA DE DUPLICADOS
# -------------------------------------------------------------------------
print(f"1. Dimensiones originales: {df.shape}")
duplicados = df.duplicated().sum()
if duplicados > 0:
    print(f"   ¡Atención! Se encontraron {duplicados} filas duplicadas.")
    df = df.drop_duplicates()
    print(f"   -> Duplicados eliminados. Nuevas dimensiones: {df.shape}")
else:
    print("   -> No se encontraron duplicados.")

# Separación de variables
X = df.drop('clase', axis=1)
y = df['clase']

# B. DIVISIÓN 80/20 (DATOS CRUDOS)
X_entr_crudo, X_prueba_crudo, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# C. NORMALIZACIÓN CORRECTA
scaler = MinMaxScaler()
scaler.fit(X_entr_crudo) # Aprende solo del train

X_entr_norm = pd.DataFrame(scaler.transform(X_entr_crudo), columns=X.columns)
X_prueba_norm = pd.DataFrame(scaler.transform(X_prueba_crudo), columns=X.columns)

# Guardar escalador
joblib.dump(scaler, 'escalador_entrenado.pkl')

print("-" * 40)
print(f">> Datos Entrenamiento (80%): {X_entr_norm.shape[0]} muestras")
print(f">> Datos Prueba Final (20%):  {X_prueba_norm.shape[0]} muestras")
print(">> Normalización completada.")
print("-" * 40)


# =============================================================================
# 4. DEFINICIÓN DE MODELOS Y PARÁMETROS
# =============================================================================
modelo_svm = SVC(probability=True, random_state=42)
params_svm = {
    'C': Continuous(0.1, 100),
    'gamma': Categorical(['scale', 'auto']),
    'kernel': Categorical(['rbf', 'linear'])
}

modelo_mlp = MLPClassifier(max_iter=1000, random_state=42)
params_mlp = {
    'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50)]),
    'activation': Categorical(['tanh', 'relu']),
    'learning_rate_init': Continuous(0.001, 0.01)
}

modelo_rf = RandomForestClassifier(random_state=42)
params_rf = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 30)
}

Lista_Modelos = [
    ("SVM", modelo_svm, params_svm),
    ("Red Neuronal", modelo_mlp, params_mlp),
    ("Random Forest", modelo_rf, params_rf)
]


# =============================================================================
# 5. EJECUCIÓN DE EXPERIMENTOS
# =============================================================================
Resultados_Totales = []

variantes = [
    ("Normalizados", "Sin Optimizar", X_entr_norm,  X_prueba_norm),
    ("Normalizados", "AG sin VC",     X_entr_norm,  X_prueba_norm),
    ("Normalizados", "AG con VC",     X_entr_norm,  X_prueba_norm)
]

print(f"{'ALGORITMO':<15} | {'ESCENARIO':<30} | {'ACCURACY':<10}")
print("-" * 65)

for nombre, modelo, params in Lista_Modelos:
    for tipo_dato, escenario, X_tr, X_te in variantes:

        if escenario == "Sin Optimizar": etiqueta = "No optimizado"
        elif escenario == "AG sin VC":   etiqueta = "Optimizado"
        else:                            etiqueta = "Optimizado + VC"

        try:
            res = ejecutar_experimento(nombre, modelo, params, X_tr, y_entrenamiento, X_te, y_prueba, escenario)
            
            res['Algoritmo'] = nombre
            res['Configuracion'] = etiqueta
            Resultados_Totales.append(res)

            print(f"{nombre:<15} | {etiqueta:<30} | {res['Accuracy']:.4f}")

        except Exception as e:
            print(f"Error en {nombre} - {escenario}: {e}")


# =============================================================================
# 6. TABLAS Y GRÁFICOS FINALES (ACTUALIZADO: TODO AUTOMATIZADO)
# =============================================================================
df_res = pd.DataFrame(Resultados_Totales)

# Limpieza de nombres
mapa_nombres = {
    "No optimizado": "Sin optimización",
    "Optimizado": "Optimizado",
    "Optimizado + VC": "Optimizado + VC"
}
df_res['Configuracion'] = df_res['Configuracion'].replace(mapa_nombres)
df_final = df_res.copy()

# Orden lógico
orden_config = ["Sin optimización", "Optimizado", "Optimizado + VC"]
df_final['Configuracion'] = pd.Categorical(df_final['Configuracion'], categories=orden_config, ordered=True)

# -----------------------------------------------------------------------------
# 1. GUARDAR CSV UNIFICADO (Métricas + Hiperparámetros)
# -----------------------------------------------------------------------------
cols_exportar = [
    'Algoritmo', 'Configuracion', 
    'Accuracy', 'Sensibilidad', 'Especificidad', 'AUC', 'VPP', 'VPN', 
    'Mejores_Parametros'
]

df_exportar = df_final[cols_exportar].sort_values(by=['Algoritmo', 'Configuracion'])

print(f"\n{'='*80}")
print("TABLA RESUMEN DE RESULTADOS")
print(f"{'='*80}")
# Mostramos en consola sin la columna de parámetros para que se lea bien
cols_vista = [c for c in cols_exportar if c != 'Mejores_Parametros']
print(df_exportar[cols_vista].round(4).to_string(index=False))

# Guardamos el archivo
df_exportar.to_csv("reporte_completo_modelos.csv", index=False)
print(f"\n>> Archivo guardado exitosamente: 'reporte_completo_modelos.csv'")

# -----------------------------------------------------------------------------
# 2. GENERAR GRÁFICOS PARA TODAS LAS MÉTRICAS
# -----------------------------------------------------------------------------
metricas_a_graficar = ['Accuracy', 'Sensibilidad', 'Especificidad', 'AUC', 'VPP', 'VPN']

print(f"\n{'='*80}")
print("GENERANDO GRÁFICOS...")
print(f"{'='*80}")

sns.set_style("whitegrid") # Estilo de fondo

for metrica in metricas_a_graficar:
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=df_final,
        x='Algoritmo',
        y=metrica,
        hue='Configuracion',
        palette='viridis'
    )
    
    # Título dinámico (Aclaración para VPP = Precisión)
    titulo = f'Comparativa de {metrica} por Modelo'
    if metrica == 'VPP': titulo += " (Precisión)"
    
    plt.title(titulo, fontsize=14, pad=20)
    plt.ylabel(f'{metrica} (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylim(0.0, 1.15)
    plt.legend(title='Estrategia', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    # Guardar imagen
    nombre_archivo = f"grafico_{metrica.lower()}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close() # Cerrar para liberar memoria
    
    print(f">> Gráfico generado: {nombre_archivo}")

print("\n¡Ejecución completada! Revisa tu carpeta para ver los resultados.")