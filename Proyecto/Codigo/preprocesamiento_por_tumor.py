import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN INICIAL
# -----------------------------------------------------------------------------
# Definición de las rutas, nombres de tumores y el tamaño de imagen.
# También se crean los directorios de salida para cada tipo de tumor.
# =============================================================================
directorio_salida = 'preprocessed_data_por_tumor'
tipos_tumor = ['Meningioma', 'Glioma', 'Adenoma_hipofisario']
TAMANO_IMAGEN = 128

print("Creando directorios de salida...")
os.makedirs(directorio_salida, exist_ok=True)
for tipo_tumor in tipos_tumor:
    directorio_tumor = os.path.join(directorio_salida, tipo_tumor)
    os.makedirs(directorio_tumor, exist_ok=True)
print("Directorios listos.")

# =============================================================================
# SECCIÓN 2: PROCESAMIENTO Y GUARDADO DE DATOS
# -----------------------------------------------------------------------------
# Este bloque localiza todos los archivos .mat, los procesa uno por uno,
# y agrupa todas las imágenes, máscaras y etiquetas por tipo de tumor.
# A diferencia del script 'por_paciente', este método mezcla datos de todos
# los pacientes, lo que puede causar "data leakage" si no se maneja con
# cuidado en la fase de entrenamiento.
# =============================================================================
directorio_fuente = '../DataSet/Tumores/'
try:
    # Obtener y ordenar la lista de archivos para un procesamiento consistente.
    lista_archivos = sorted(os.listdir(directorio_fuente))
except FileNotFoundError:
    print(f"Error: El directorio del dataset no fue encontrado en '{directorio_fuente}'")
    print("Por favor, asegúrate de que la ruta al dataset es correcta.")
    exit()

# Se inicializa un diccionario para almacenar los datos.
# Cada tipo de tumor (identificado por su etiqueta 1, 2, o 3) tendrá
# su propia lista de imágenes, máscaras y etiquetas.
datos_por_tumor = {
    1: {'imagenes': [], 'mascaras': [], 'etiquetas': []}, # Meningioma
    2: {'imagenes': [], 'mascaras': [], 'etiquetas': []}, # Glioma
    3: {'imagenes': [], 'mascaras': [], 'etiquetas': []}  # Adenoma Hipofisario
}
contador_errores = 0

print(f"Procesando {len(lista_archivos)} archivos...")
for nombre_archivo in tqdm(lista_archivos, desc="Procesando archivos"):
    try:
        ruta_completa = os.path.join(directorio_fuente, nombre_archivo)
        with h5py.File(ruta_completa, 'r') as archivo_h5:
            etiqueta = int(archivo_h5['cjdata']['label'][0][0])
            
            # Si la etiqueta no es una de las que nos interesan (1, 2, 3), saltamos el archivo.
            if etiqueta not in datos_por_tumor:
                contador_errores += 1
                continue

            # Extracción de la imagen y la máscara del tumor.
            imagen = np.array(archivo_h5['cjdata']['image'])
            mascara = np.array(archivo_h5['cjdata']['tumorMask'])

            # Redimensionamiento de imagen y máscara.
            imagen_redimensionada = cv2.resize(imagen, (TAMANO_IMAGEN, TAMANO_IMAGEN), interpolation=cv2.INTER_CUBIC)
            mascara_redimensionada = cv2.resize(mascara, (TAMANO_IMAGEN, TAMANO_IMAGEN), interpolation=cv2.INTER_NEAREST)

            # Conversión de tipos de datos.
            imagen_a_guardar = imagen_redimensionada.astype(np.float32)
            mascara_a_guardar = mascara_redimensionada.astype(np.uint8)

            # ** CAMBIO IMPORTANTE: NO se aplica la máscara a la imagen. **
            # Se guardan la imagen y la máscara por separado para dar más flexibilidad
            # durante el entrenamiento del modelo.

            # Añadir los datos procesados a las listas correspondientes.
            datos_por_tumor[etiqueta]['imagenes'].append(imagen_a_guardar)
            datos_por_tumor[etiqueta]['mascaras'].append(mascara_a_guardar)
            datos_por_tumor[etiqueta]['etiquetas'].append(etiqueta)

    except Exception as e:
        # print(f"Error procesando el archivo {nombre_archivo}: {e}") # Descomentar para depuración
        contador_errores += 1
        continue

# =============================================================================
# SECCIÓN 3: GUARDADO DE ARCHIVOS PREPROCESADOS
# -----------------------------------------------------------------------------
# Para cada tipo de tumor, se empaquetan todas las imágenes, máscaras y
# etiquetas en un único archivo .npz comprimido. El resultado es un archivo
# grande por cada tipo de tumor.
# =============================================================================
print("\nGuardando archivos preprocesados...")
for id_etiqueta, nombre_tumor in enumerate(tipos_tumor, 1):
    directorio_tumor = os.path.join(directorio_salida, nombre_tumor)
    
    # Extraer los datos del diccionario para el tumor actual.
    datos_grupo = datos_por_tumor[id_etiqueta]
    
    if len(datos_grupo['imagenes']) > 0:
        ruta_salida = os.path.join(directorio_tumor, f'{nombre_tumor}_preprocesado.npz')
        
        # Guardar los tres arreglos (imágenes, máscaras, etiquetas) en el archivo.
        np.savez_compressed(
            ruta_salida, 
            imagenes=np.array(datos_grupo['imagenes']), 
            mascaras=np.array(datos_grupo['mascaras']),
            etiquetas=np.array(datos_grupo['etiquetas'])
        )
        print(f'Se guardaron {len(datos_grupo["imagenes"])} imágenes para {nombre_tumor} en {ruta_salida}')
    else:
        print(f"No se encontraron imágenes para el tumor {nombre_tumor}.")

print(f"\nProceso completado.")
print(f"Total de archivos con errores o saltados: {contador_errores}")

