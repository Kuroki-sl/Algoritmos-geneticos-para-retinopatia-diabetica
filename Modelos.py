# =============================================================================
# 6. TABLAS Y GRÁFICOS FINALES (ACTUALIZADO: CSV UNIFICADO + TODOS LOS GRÁFICOS)
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. PREPARACIÓN DE DATOS
# ------------------------------------------------------
df_res = pd.DataFrame(Resultados_Totales)

# Diccionario para filtrar y renombrar configuraciones
mapa_nombres = {
    "No optimizado": "Sin optimización",
    "Optimizado": "Optimizado",
    "Optimizado + VC": "Optimizado + VC"
}

# Aplicar nombres limpios
df_res['Configuracion'] = df_res['Configuracion'].replace(mapa_nombres)
df_final = df_res.copy()

# Definir el orden lógico de las barras
orden_config = ["Sin optimización", "Optimizado", "Optimizado + VC"]
df_final['Configuracion'] = pd.Categorical(df_final['Configuracion'], categories=orden_config, ordered=True)

# -----------------------------------------------------------------------------
# REQUERIMIENTO 1: GUARDAR UN ÚNICO CSV CON TODO (MÉTRICAS + HIPERPARÁMETROS)
# -----------------------------------------------------------------------------
# Seleccionamos y ordenamos las columnas para que sea legible
cols_exportar = [
    'Algoritmo', 
    'Configuracion', 
    'Accuracy', 
    'Sensibilidad', 
    'Especificidad', 
    'AUC', 
    'VPP', 
    'VPN', 
    'Mejores_Parametros' # Aquí incluimos los hiperparámetros en la misma tabla
]

# Creamos el DataFrame final ordenado
df_exportar = df_final[cols_exportar].sort_values(by=['Algoritmo', 'Configuracion'])

# Mostramos un adelanto en consola (sin la columna de parámetros para que no se desordene la vista)
print(f"\n{'='*80}")
print("TABLA RESUMEN DE MÉTRICAS")
print(f"{'='*80}")
cols_vista = [c for c in cols_exportar if c != 'Mejores_Parametros']
print(df_exportar[cols_vista].round(4).to_string(index=False))

# Guardamos el archivo ÚNICO
nombre_csv = "reporte_completo_modelos.csv"
df_exportar.to_csv(nombre_csv, index=False)
print(f"\n>> Se ha generado el archivo unificado: '{nombre_csv}'")


# -----------------------------------------------------------------------------
# REQUERIMIENTO 2: GENERACIÓN DE GRÁFICOS PARA TODAS LAS MÉTRICAS
# -----------------------------------------------------------------------------
# Lista completa de métricas a graficar
# Nota: 'Presisión' es matemáticamente igual al VPP, por lo que el gráfico de VPP cubre ambos.
metricas_a_graficar = ['Accuracy', 'Sensibilidad', 'Especificidad', 'AUC', 'VPP', 'VPN']

print(f"\n{'='*80}")
print("GENERANDO GRÁFICOS...")
print(f"{'='*80}")

# Configuración estética general
sns.set_style("whitegrid")

for metrica in metricas_a_graficar:
    # Crear una figura nueva para cada métrica
    plt.figure(figsize=(10, 6))
    
    # Crear el gráfico de barras
    grafico = sns.barplot(
        data=df_final,
        x='Algoritmo',
        y=metrica,
        hue='Configuracion',
        palette='viridis'
    )
    
    # Títulos y etiquetas dinámicas
    titulo = f'Comparativa de {metrica} por Modelo'
    if metrica == 'VPP': titulo += " (Precisión)" # Aclaración visual
    
    plt.title(titulo, fontsize=14, pad=20)
    plt.ylabel(f'{metrica} (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Algoritmo', fontsize=12)
    plt.ylim(0.0, 1.15) # Un poco más de espacio arriba para la leyenda
    
    # Leyenda ubicada fuera para no tapar barras
    plt.legend(title='Estrategia', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar el archivo con nombre dinámico
    nombre_archivo = f"grafico_{metrica.lower()}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    
    # Cerrar la figura para liberar memoria
    plt.close()
    
    print(f">> Gráfico generado: {nombre_archivo}")

print(f"\n¡Proceso finalizado! Revisa la carpeta para ver el CSV y las {len(metricas_a_graficar)} imágenes.")