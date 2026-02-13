# Implementación del Proyecto

Esta carpeta contiene todos los recursos relacionados con la implementación del modelo híbrido cuántico-clásico para la clasificación de tumores cerebrales.

## Contenido

- **`/Codigo`**: Contiene los notebooks de Jupyter (`.ipynb`) y scripts de Python (`.py`) con el código del proyecto. Aquí se realizarán la exploración de datos, el preprocesamiento, la implementación del modelo y la evaluación de resultados.
- **`/DataSet`**: Contiene el conjunto de datos de imágenes de tumores cerebrales en formato `.mat`.

## Metodología

El flujo de trabajo seguirá los siguientes pasos:

1.  **Carga y Exploración de Datos**: Cargar las imágenes `.mat` y analizar su estructura.
2.  **Preprocesamiento**: Segmentar la región de interés usando la máscara del tumor, redimensionar y normalizar las imágenes.
3.  **Embedding Cuántico**: Implementar el codificador de amplitudes (`Amplitude Encoding`) usando PennyLane para transformar los datos de las imágenes en estados cuánticos.
4.  **Modelo Clásico**: Construir una red neuronal convolucional (CNN) en TensorFlow o PyTorch para clasificar los vectores de características extraídos del circuito cuántico.
5.  **Entrenamiento y Evaluación**: Entrenar el modelo híbrido y evaluar su rendimiento utilizando métricas como precisión, sensibilidad y la matriz de confusión.
