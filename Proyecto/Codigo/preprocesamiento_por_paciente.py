import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# =============================================================================
# SECCIÓN 1: CONFIGURACIÓN INICIAL
# -----------------------------------------------------------------------------
# En esta sección se definen las variables principales que controlan el script,
# como las rutas de los directorios, los tipos de tumores a procesar y el
# tamaño al que se redimensionarán las imágenes.
# =============================================================================
directorio_salida = 'Preprocesamiento_por_paciente'
directorio_fuente = '../DataSet/Tumores/' 
tipos_tumor = ['Meningioma', 'Glioma', 'Adenoma_hipofisario']
TAMANO_IMAGEN = 128

# Se crean los directorios de salida para cada tipo de tumor si no existen.
# Esto asegura que tengamos dónde guardar los archivos procesados.
print("Creando directorios de salida...")
for tipo in tipos_tumor:
    os.makedirs(os.path.join(directorio_salida, tipo), exist_ok=True)
print("Directorios listos.")

# =============================================================================
# SECCIÓN 2: FUNCIÓN DE DECODIFICACIÓN
# -----------------------------------------------------------------------------
# El ID del paciente (PID) en los archivos .mat originales no es un texto
# simple, sino un arreglo de números (códigos ASCII). Esta función toma
# ese arreglo, lo convierte a caracteres y devuelve el ID como una cadena
# de texto legible.
# =============================================================================
def decodificar_id_paciente(datos_id):
    """
    Decodifica el ID de paciente desde el formato de arreglo de los archivos .mat.
    """
    try:
        # Aplanar el arreglo y filtrar valores nulos o no válidos.
        arreglo = np.array(datos_id).flatten()
        if arreglo.size > 1:
            # Convertir cada código ASCII a su caracter correspondiente.
            caracteres = [chr(int(x)) for x in arreglo if x > 0]
            return "".join(caracteres).strip()
        elif arreglo.size == 1:
            return str(arreglo[0])
        return "Desconocido"
    except Exception:
        return "Error"

# =============================================================================
# SECCIÓN 3: BÚSQUEDA DE ARCHIVOS DE DATOS
# -----------------------------------------------------------------------------
# Antes de procesar, necesitamos encontrar todos los archivos de datos.
# Este bloque de código recorre el directorio fuente de forma recursiva
# para encontrar todos los archivos con la extensión .mat y los añade a una
# lista para su posterior procesamiento.
# =============================================================================
print(f"Buscando archivos .mat en '{directorio_fuente}'...")
rutas_archivos_mat = []
for raiz, _, archivos in os.walk(directorio_fuente):
    for nombre_archivo in archivos:
        if nombre_archivo.endswith('.mat'):
            rutas_archivos_mat.append(os.path.join(raiz, nombre_archivo))

print(f"Se encontraron {len(rutas_archivos_mat)} archivos para procesar.")

# Diccionarios para agrupar los datos por paciente antes de guardarlos.
datos_por_paciente = {}
etiquetas_por_paciente = {}
contador_errores = 0

# =============================================================================
# SECCIÓN 4: PROCESAMIENTO DE IMÁGENES
# -----------------------------------------------------------------------------
# Este es el corazón del script. Itera sobre cada archivo .mat encontrado,
# extrae la imagen, la máscara del tumor y la etiqueta. Luego, redimensiona
# la imagen y la máscara y las agrupa por paciente en un diccionario.
# Este enfoque ("por paciente") es crucial para evitar el "data leakage"
# en el futuro, asegurando que todas las imágenes de un paciente queden
# juntas.
# =============================================================================
print("Procesando y agrupando imágenes por paciente...")
for ruta_archivo in tqdm(rutas_archivos_mat, desc="Procesando archivos"):
    try:
        # Lectura de datos desde el archivo .mat
        with h5py.File(ruta_archivo, 'r') as archivo_h5:
            id_paciente_raw = archivo_h5['cjdata']['PID']
            id_paciente = decodificar_id_paciente(id_paciente_raw)
            
            # Si el ID no se pudo decodificar, se salta el archivo.
            if id_paciente in ["Error", "Desconocido"]:
                contador_errores += 1
                continue

            etiqueta = int(np.array(archivo_h5['cjdata']['label']).flatten()[0])
            imagen = np.array(archivo_h5['cjdata']['image'])
            mascara = np.array(archivo_h5['cjdata']['tumorMask'])

        # Redimensionamiento de la imagen y la máscara
        # Se usa interpolación CÚBICA para la imagen para preservar más detalles.
        imagen_redimensionada = cv2.resize(imagen, (TAMANO_IMAGEN, TAMANO_IMAGEN), interpolation=cv2.INTER_CUBIC)
        # Se usa interpolación NEAREST para la máscara para mantener sus valores binarios (0 o 1).
        mascara_redimensionada = cv2.resize(mascara, (TAMANO_IMAGEN, TAMANO_IMAGEN), interpolation=cv2.INTER_NEAREST)

        # Conversión de tipos de datos para optimizar el almacenamiento
        # La imagen se convierte a float32 para estandarizarla.
        imagen_a_guardar = imagen_redimensionada.astype(np.float32)
        # La máscara se convierte a uint8 (0-255) para ahorrar espacio.
        mascara_a_guardar = mascara_redimensionada.astype(np.uint8)

        # Agrupación de datos por paciente
        # Si es la primera vez que vemos a este paciente, creamos una nueva entrada en los diccionarios.
        if id_paciente not in datos_por_paciente:
            datos_por_paciente[id_paciente] = {'imagenes': [], 'mascaras': []}
            etiquetas_por_paciente[id_paciente] = etiqueta
        
        # Añadimos la imagen y máscara procesadas a la lista de ese paciente.
        datos_por_paciente[id_paciente]['imagenes'].append(imagen_a_guardar)
        datos_por_paciente[id_paciente]['mascaras'].append(mascara_a_guardar)

    except Exception as e:
        # print(f"Error procesando {ruta_archivo}: {e}") # Descomentar para depuración
        contador_errores += 1
        continue

# =============================================================================
# SECCIÓN 5: GUARDADO DE DATOS
# -----------------------------------------------------------------------------
# Una vez que todas las imágenes han sido procesadas y agrupadas, este bloque
# itera sobre cada paciente y guarda todas sus imágenes, máscaras y su
# etiqueta correspondiente en un único archivo comprimido .npz.
# El nombre del archivo incluye el ID del paciente para una fácil identificación.
# =============================================================================
print(f"Procesamiento finalizado. Guardando datos de {len(datos_por_paciente)} pacientes...")
for id_paciente, datos in tqdm(datos_por_paciente.items(), desc="Guardando pacientes"):
    etiqueta = etiquetas_por_paciente[id_paciente]
    
    # Asegurarse de que la etiqueta es válida (1, 2, o 3)
    if 1 <= etiqueta <= len(tipos_tumor):
        # El label es 1-indexed, pero la lista de tipos de tumor es 0-indexed.
        # Por eso se resta 1 para obtener el nombre correcto del directorio.
        directorio_destino = os.path.join(directorio_salida, tipos_tumor[etiqueta - 1])
        ruta_guardado = os.path.join(directorio_destino, f'paciente_{id_paciente}.npz')
        
        # Guardar los datos en un archivo .npz comprimido.
        # np.savez_compressed es eficiente para guardar múltiples arreglos de numpy.
        np.savez_compressed(
            ruta_guardado, 
            imagenes=np.array(datos['imagenes']), 
            mascaras=np.array(datos['mascaras']), 
            etiqueta=etiqueta
        )
    else:
        print(f"ADVERTENCIA: Se omitió el paciente {id_paciente} con etiqueta inválida: {etiqueta}")

print(f"\nProceso completado.")
print(f"Total de pacientes guardados: {len(datos_por_paciente)}")
print(f"Total de archivos con errores: {contador_errores}")