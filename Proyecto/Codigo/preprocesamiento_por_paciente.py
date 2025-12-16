import os
import h5py
import scipy.io
import numpy as np
import cv2
from tqdm import tqdm

# --- CONFIGURACIÓN ---
base_dir = 'dataset_por_paciente_final'
source_folder = '../DataSet/Tumores/' 
tumor_types = ['Meningioma', 'Glioma', 'Adenoma_hipofisario']
IMG_SIZE = 128

# Crear directorios
for t in tumor_types:
    os.makedirs(os.path.join(base_dir, t), exist_ok=True)

# --- FUNCIÓN DE DECODIFICACIÓN (La que funcionó) ---
def decode_pid(pid_data):
    try:
        arr = np.array(pid_data).flatten()
        if arr.size > 1:
            arr = arr[arr > 0]
            chars = [chr(int(x)) for x in arr]
            return "".join(chars).strip()
        elif arr.size == 1:
            return str(arr[0])
        return "Unknown"
    except:
        return "Error"

# 1. BÚSQUEDA DE ARCHIVOS
print(f"Buscando archivos en {source_folder}...")
archivos_mat = []
for root, _, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.mat'):
            archivos_mat.append(os.path.join(root, file))

print(f"Archivos encontrados: {len(archivos_mat)}")

dataPorPaciente = {}
labelsPorPaciente = {}
errores = 0

# 2. PROCESAMIENTO
print("Procesando imágenes...")
for ruta in tqdm(archivos_mat):
    try:
        pid_final = None
        label = None
        image = None
        mask = None

        # Intento 1: H5PY
        try:
            with h5py.File(ruta, 'r') as f:
                pid_raw = f['cjdata']['PID']
                pid_final = decode_pid(pid_raw)
                label = int(np.array(f['cjdata']['label']).flatten()[0])
                image = np.array(f['cjdata']['image'])
                mask = np.array(f['cjdata']['tumorMask'])
        except (OSError, KeyError):
            # Intento 2: Scipy
            mat = scipy.io.loadmat(ruta)
            cjdata = mat['cjdata']
            pid_raw = cjdata['PID'][0][0]
            pid_final = decode_pid(pid_raw)
            label_elem = cjdata['label'][0][0]
            label = int(label_elem.flatten()[0]) if isinstance(label_elem, np.ndarray) else int(label_elem)
            image = cjdata['image'][0][0]
            mask = cjdata['tumorMask'][0][0]

        if pid_final == "Error" or pid_final == "Unknown":
            errores += 1
            continue

        # Resize
        # Imagen: Cúbica para mantener detalles
        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        # Máscara: Nearest para mantener binario puro
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Cast a uint8 para ahorrar espacio (máscara)
        # La imagen se deja en su tipo original (probablemente int16 o float) o se convierte si es necesario
        # Aquí convertimos la imagen a float32 para estandarizar sin normalizar rango
        image_save = image_resized.astype(np.float32)
        mask_save = mask_resized.astype(np.uint8)

        # Agrupar
        if pid_final not in dataPorPaciente:
            dataPorPaciente[pid_final] = {'images': [], 'masks': []}
            labelsPorPaciente[pid_final] = label
        
        dataPorPaciente[pid_final]['images'].append(image_save)
        dataPorPaciente[pid_final]['masks'].append(mask_save)

    except Exception as e:
        errores += 1
        continue

# 3. GUARDADO
print(f"Guardando {len(dataPorPaciente)} pacientes...")
for pid, data in dataPorPaciente.items():
    label = labelsPorPaciente[pid]
    # Restar 1 al label porque el array es 0-indexed (Label 1 -> Index 0)
    if 1 <= label <= 3:
        target_dir = os.path.join(base_dir, tumor_types[label - 1])
        save_path = os.path.join(target_dir, f'patient_{pid}.npz')
        
        np.savez_compressed(
            save_path, 
            images=np.array(data['images']), 
            masks=np.array(data['masks']), 
            label=label
        )
    else:
        print(f"Skipping PID {pid} con label inválido: {label}")

print(f"Proceso finalizado. Total Errores: {errores}")