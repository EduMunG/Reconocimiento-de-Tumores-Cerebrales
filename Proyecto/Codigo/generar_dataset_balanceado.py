import os
import glob
import numpy as np
import shutil
from collections import defaultdict
from tqdm import tqdm

"""
SCRIPT DE GENERACIÓN DE DATASET BALANCEADO PARA MODELO HÍBRIDO CUÁNTICO

PROPÓSITO:
Este script toma el dataset preprocesado (organizado por pacientes) y genera una nueva versión balanceada
donde cada clase (Meningioma, Glioma, Adenoma) tiene exactamente el mismo número de imágenes (TARGET_COUNT).

¿POR QUÉ ES NECESARIO?
1. Eficiencia Cuántica: Los circuitos cuánticos son costosos computacionalmente. Reducir el dataset a un tamaño
   manejable y balanceado es crucial para que el entrenamiento sea viable en tiempo.
2. Consistencia: Al guardar este dataset en disco, aseguramos que todos los experimentos (clásicos o cuánticos)
   usen exactamente las mismas imágenes, garantizando reproducibilidad.
3. Integridad Médica: A diferencia de un borrado aleatorio, este script usa 'Submuestreo Uniforme'.
   Si un paciente tiene 30 cortes y solo necesitamos 10, seleccionamos cortes espaciados uniformemente
   (ej. inicio, medio, final) para preservar la estructura volumétrica del tumor.
"""

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
INPUT_DIR = 'Preprocesamiento_por_paciente'  # Directorio fuente con todos los datos
OUTPUT_DIR = 'Dataset_Balanceado_700'        # Directorio destino para el dataset limpio
TARGET_COUNT = 700                           # Número de imágenes objetivo por clase
CLASES = ['Meningioma', 'Glioma', 'Adenoma_hipofisario']

def crear_dataset_balanceado():
    """
    Función principal que orquesta la lectura, filtrado y guardado del nuevo dataset.
    """
    
    # Ajuste automático de rutas dependiendo de dónde se ejecute el script (raíz o carpeta Codigo)
    global INPUT_DIR, OUTPUT_DIR
    if not os.path.exists(INPUT_DIR):
        posible_path = os.path.join('Codigo', INPUT_DIR)
        if os.path.exists(posible_path):
            INPUT_DIR = posible_path
            OUTPUT_DIR = os.path.join('Codigo', OUTPUT_DIR)
        elif os.path.exists(os.path.join('..', INPUT_DIR)):
             INPUT_DIR = os.path.join('..', INPUT_DIR)
             OUTPUT_DIR = os.path.join('..', OUTPUT_DIR)
            
    if not os.path.exists(INPUT_DIR):
        print(f"Error Crítico: No se encontró el directorio de entrada '{INPUT_DIR}'.")
        print("Ejecuta primero 'preprocesamiento_por_paciente.py' para generar los datos base.")
        return

    # Preparar directorio de salida limpio
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Borrar si existe para empezar de cero
    os.makedirs(OUTPUT_DIR)

    print(f"--- INICIANDO GENERACIÓN DE DATASET BALANCEADO ---")
    print(f"Fuente: {INPUT_DIR}")
    print(f"Destino: {OUTPUT_DIR}")
    print(f"Objetivo: {TARGET_COUNT} imágenes por clase.")
    print(f"Método: Selección Uniforme con Compensación de Error (Accumulator).")

    for clase in CLASES:
        # Rutas específicas por clase
        input_path = os.path.join(INPUT_DIR, clase)
        output_path_class = os.path.join(OUTPUT_DIR, clase)
        os.makedirs(output_path_class, exist_ok=True)
        
        # ---------------------------------------------------------------------
        # PASO 1: Inventario de Pacientes
        # ---------------------------------------------------------------------
        # Agrupamos los archivos .npz por ID de paciente para tratarlos como una unidad.
        files_by_patient = defaultdict(list)
        all_files = glob.glob(os.path.join(input_path, '*.npz'))
        
        for f in all_files:
            filename = os.path.basename(f)
            # Formato esperado: paciente_ID.npz
            pid = filename.replace('paciente_', '').replace('.npz', '')
            files_by_patient[pid].append(f)
            
        # Ordenamos los archivos de cada paciente para asegurar consistencia
        for pid in files_by_patient:
            files_by_patient[pid].sort()

        # ---------------------------------------------------------------------
        # PASO 2: Análisis Estadístico y Cálculo de Ratio
        # ---------------------------------------------------------------------
        total_images_class = 0
        patient_image_counts = {} # Mapa: ID_Paciente -> Número de cortes
        
        print(f"\nAnalizando clase: {clase}...")
        
        # Leemos solo los metadatos (shape) para contar sin cargar todo a memoria RAM
        for pid, files in files_by_patient.items():
            if len(files) > 0:
                try:
                    with np.load(files[0]) as data:
                        # Buscamos la clave correcta (a veces guardada como 'imagenes' o 'images')
                        key = 'imagenes' if 'imagenes' in data else 'images'
                        n = data[key].shape[0]
                        patient_image_counts[pid] = n
                        total_images_class += n
                except Exception as e:
                    print(f"  [Error] No se pudo leer {files[0]}: {e}")
                    patient_image_counts[pid] = 0

        total_patients = len(files_by_patient)
        print(f"  - Inventario: {total_images_class} imágenes totales de {total_patients} pacientes.")
        
        # Determinamos qué porcentaje de imágenes debemos conservar
        if total_images_class <= TARGET_COUNT:
            print("  - Estado: Déficit o exacto. Se copiarán TODAS las imágenes.")
            keep_ratio = 1.0
        else:
            keep_ratio = TARGET_COUNT / total_images_class
            print(f"  - Estado: Exceso. Se aplicará reducción del {100 - (keep_ratio*100):.2f}% (Ratio: {keep_ratio:.4f})")

        # ---------------------------------------------------------------------
        # PASO 3: Selección Inteligente y Guardado
        # ---------------------------------------------------------------------
        count_saved = 0
        accumulator = 0.0  # Variable crítica para "Difusión de Error" y lograr el número exacto
        
        # Barra de progreso para visualización
        pbar = tqdm(files_by_patient.items(), desc=f"  Procesando {clase}")
        
        for pid, files in pbar:
            if len(files) == 0: continue
            
            f_path = files[0]
            n_slices = patient_image_counts.get(pid, 0)
            
            if n_slices == 0: continue

            # --- LÓGICA DEL ACUMULADOR ---
            # Si simplemente hacemos int(n * ratio), perdemos los decimales (ej. 0.7 imágenes).
            # Al sumar esos decimales perdidos en 'accumulator', cuando juntamos 1.0,
            # seleccionamos una imagen extra. Esto garantiza que la suma total sea exacta.
            
            if keep_ratio >= 1.0:
                selected_indices = range(n_slices)
            else:
                exact_to_keep = n_slices * keep_ratio
                n_keep_slices = int(exact_to_keep)
                
                # Guardamos la parte decimal (el "error" de redondeo)
                remainder = exact_to_keep - n_keep_slices
                accumulator += remainder
                
                # Si acumulamos suficiente error, salvamos una imagen extra
                if accumulator >= 1.0:
                    n_keep_slices += 1
                    accumulator -= 1.0
                
                # Límites de seguridad
                if n_keep_slices < 1 and n_slices > 0: n_keep_slices = 1 # Intentar no dejar pacientes vacíos
                if n_keep_slices > n_slices: n_keep_slices = n_slices    # No pedir más de lo que hay
                
                # Selección Equiespaciada (np.linspace)
                # Selecciona índices distribuidos uniformemente (ej: 0, 5, 10...)
                # para cubrir todo el volumen del cerebro, no solo un segmento.
                if n_keep_slices > 0:
                    selected_indices = np.linspace(0, n_slices - 1, n_keep_slices, dtype=int)
                    selected_indices = np.unique(selected_indices)
                else:
                    selected_indices = []

            # --- GUARDADO ---
            if len(selected_indices) > 0:
                try:
                    data = np.load(f_path)
                    imgs_all = data['imagenes'] if 'imagenes' in data else data['images']
                    masks_all = data['mascaras'] if 'mascaras' in data else data['masks']
                    
                    # Recuperar etiqueta original
                    if 'etiqueta' in data:
                        label = data['etiqueta']
                    elif 'label' in data:
                        label = data['label']
                    else:
                        label = CLASES.index(clase) + 1 

                    # Crear subconjunto filtrado
                    imgs_final = imgs_all[selected_indices]
                    masks_final = masks_all[selected_indices]
                    
                    # Guardar nuevo archivo comprimido .npz
                    new_filename = os.path.basename(f_path)
                    save_path = os.path.join(output_path_class, new_filename)
                    
                    np.savez_compressed(
                        save_path,
                        imagenes=imgs_final,
                        mascaras=masks_final,
                        etiqueta=label
                    )
                    count_saved += len(imgs_final)
                        
                except Exception as e:
                    print(f"  [Error] Fallo al procesar archivo {f_path}: {e}")

        print(f"  -> Resultado Final: {count_saved} imágenes guardadas para {clase}.")

    print("\n--- PROCESO COMPLETADO EXITOSAMENTE ---")
    print(f"El dataset balanceado se encuentra en: {OUTPUT_DIR}")

if __name__ == "__main__":
    crear_dataset_balanceado()
