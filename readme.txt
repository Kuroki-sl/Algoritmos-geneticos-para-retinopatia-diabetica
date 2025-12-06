Optimización de Modelos de IA para Detección de Retinopatía Diabética

Este proyecto explora y compara el rendimiento de diferentes algoritmos de clasificación (SVM, Redes Neuronales y Random Forest) aplicados al dataset "messidor_features.arff". El objetivo principal es mejorar las métricas de predicción mediante la optimización de hiperparámetros utilizando Algoritmos Genéticos.

Descripción del Proyecto:

El análisis busca clasificar pacientes basándose en características extraídas de imágenes de la retina. Se evalúa el impacto del preprocesamiento de datos y la búsqueda evolutiva de hiperparámetros frente a los modelos con configuración por defecto.

El flujo de trabajo incluye:
1.Preprocesamiento: Limpieza y normalización (MinMaxScaler) de los datos.
2.División de Datos: 80% Entrenamiento/Validación y 20% Prueba Final (Hold-out).
3.Modelos Evaluados:
    Support Vector Machine (SVM)
    Perceptrón Multicapa (MLP/Red Neuronal)
    Random Forest
4.Optimización: Uso de la librería "sklearn-genetic-opt" para búsqueda de hiperparámetros.

Estructura del Repositorio:

"Modelos.py": Codigo principal para el entrenamiento, optimización y visualización.
"messidor_features.arff": Dataset utilizado (debe estar en la raíz para ejecutar el código).
"requirements.txt": Lista de dependencias necesarias.

El ejecutar el codigo generará los siguientes elementos:

"graficos": una carpeta que contendra todos las imagenes de los graficos de las metricas calculadas.
"escalador_entrenado.pkl": archivo que guardara los parametros de normalizacion que uso para el 80% que se uso para el entrenamiento y los usara para normalizar el 20% que el modelo no "conoce"
"Metricas.csv": archivo que contendra las metricas e hiperparámetros que obtuvo el codigo al ejecutarse.

Requisitos e Instalación:

Este proyecto utiliza Python 3. Para instalar las librerías necesarias, ejecuta:

bash: pip install -r requirements.txt
