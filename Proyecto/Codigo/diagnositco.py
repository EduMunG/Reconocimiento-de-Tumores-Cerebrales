import os
import h5py
import scipy.io
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CONFIGURACIÓN ---
source_folder = '../DataSet/Tumores/'  # Ruta al dataset

print(f"Escanendo archivos en: {source_folder}")

records = []
errores_lectura = 0

# 1. BARRIDO DE ARCHIVOS
files_to_process = []
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.mat'):
            files_to_process.append(os.path.join(root, file))

print(f"Total archivos .mat detectados: {len(files_to_process)}")

# --- FUNCIÓN DE DECODIFICACIÓN ROBUSTA ---
def decode_pid(pid_data):
    """Convierte array de enteros (formato MATLAB string) a string Python"""
    try:
        arr = np.array(pid_data).flatten()
        
        # Caso 1: Es un array de caracteres/enteros (e.g. [49, 48, 50]) -> "102"
        if arr.size > 1:
            # Filtrar ceros o valores nulos si existen
            arr = arr[arr > 0]
            # Convertir enteros a caracteres ASCII
            chars = [chr(int(x)) for x in arr]
            return "".join(chars).strip()
            
        # Caso 2: Es un escalar único
        elif arr.size == 1:
            val = arr[0]
            # Si es un número pequeño (ej. < 255), podría ser un char aislado
            # Pero asumimos que si es escalar, es el ID numérico directo
            return str(val)
            
        return "Unknown"
    except Exception:
        return "Error"

# 2. EXTRACCIÓN DE METADATOS
for file_path in tqdm(files_to_process):
    filename = os.path.basename(file_path)
    pid_final = "Error"
    label_raw = None
    read_method = "None"
    
    try:
        # Intento 1: H5PY (Formato v7.3+)
        with h5py.File(file_path, 'r') as f:
            pid_data = f['cjdata']['PID']
            label_data = f['cjdata']['label']
            
            # Decodificar PID usando la nueva función
            pid_final = decode_pid(pid_data)
            label_raw = np.array(label_data).flatten()[0]
            read_method = "h5py"

    except (OSError, KeyError):
        # Intento 2: Scipy (Formato v7.2 o menor)
        try:
            mat = scipy.io.loadmat(file_path)
            cjdata = mat['cjdata']
            
            pid_elem = cjdata['PID'][0][0]
            label_elem = cjdata['label'][0][0]
            
            pid_final = decode_pid(pid_elem)
            label_raw = label_elem.flatten()[0] if isinstance(label_elem, np.ndarray) else label_elem
            read_method = "scipy"
            
        except Exception as e:
            errores_lectura += 1
            read_method = "error"

    records.append({
        "Filename": filename,
        "PID_Corrected": pid_final,
        "Label": label_raw,
        "Method": read_method,
        "Path": file_path
    })

# 3. ANÁLISIS CON PANDAS
df = pd.DataFrame(records)

print("\n" + "="*40)
print("REPORTE DE DIAGNÓSTICO (V2 - DECODED)")
print("="*40)

valid_df = df[df['Method'] != 'error']

if valid_df.empty:
    print("CRÍTICO: No se pudo leer ningún archivo correctamente.")
else:
    unique_pids = valid_df['PID_Corrected'].unique()
    count_pids = len(unique_pids)
    
    print(f"Archivos leídos OK: {len(valid_df)}")
    print(f"PIDs Únicos encontrados: {count_pids}")
    
    print("\n--- Muestra de PIDs encontrados (Primeros 15) ---")
    print(sorted(unique_pids)[:15])
    
    # Validación de integridad
    if count_pids >= 200:
        print(f"\n✅ ÉXITO: Se recuperaron {count_pids} pacientes (Cercano a los 233 esperados).")
    else:
        print(f"\n⚠️ ALERTA: Aún hay pocos pacientes ({count_pids}). Revisa el CSV.")

    # Guardar CSV corregido
    csv_path = "reporte_dataset_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"Reporte guardado en: {csv_path}")