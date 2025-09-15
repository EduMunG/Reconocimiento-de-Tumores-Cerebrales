import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# Creación de directorios
base_dir = 'preprocessed_data_por_tumor'
tumor_types = ['Meningioma', 'Glioma', 'Adenoma_hipofisario']

print("Creando directorios...")
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for tumor_type in tumor_types:
    tumor_dir = os.path.join(base_dir, tumor_type)
    if not os.path.exists(tumor_dir):
        os.makedirs(tumor_dir)
print("Directorios creados.")

# Procesamiento y guardado de datos
folder = '../DataSet/Tumores/'
try:
    lista = sorted(os.listdir(folder))
except FileNotFoundError:
    print(f"Error: El directorio del dataset no fue encontrado en '{folder}'")
    print("Por favor, asegúrate de que la ruta al dataset es correcta.")
    exit()

data_by_tumor = {1: [], 2: [], 3: []}
labels_by_tumor = {1: [], 2: [], 3: []}

print(f"Procesando {len(lista)} archivos...")
for archivo in tqdm(lista):
    try:
        with h5py.File(os.path.join(folder, archivo), 'r') as data:
            label = int(data['cjdata']['label'][0][0])
            if label not in data_by_tumor:
                continue

            image = np.array(data['cjdata']['image'])
            mask = np.array(data['cjdata']['tumorMask'])

            # Redimensionar imagen y máscara a 128x128
            image_resized = cv2.resize(image, (128, 128))
            mask_resized = cv2.resize(mask, (128, 128))

            # Normalizar imagen
            image_normalized = cv2.normalize(image_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

            # Aplicar máscara
            tumor_image = image_normalized * mask_resized

            data_by_tumor[label].append(tumor_image)
            labels_by_tumor[label].append(label)
    except Exception as e:
        print(f"Error procesando el archivo {archivo}: {e}")

print("Guardando archivos preprocesados...")
for label_id, tumor_name in enumerate(tumor_types, 1):
    tumor_dir = os.path.join(base_dir, tumor_name)
    images = np.array(data_by_tumor[label_id])
    labels = np.array(labels_by_tumor[label_id])
    
    if images.size > 0:
        output_path = os.path.join(tumor_dir, f'{tumor_name}_preprocessed.npz')
        np.savez_compressed(output_path, images=images, labels=labels)
        print(f'Se guardaron {len(images)} imágenes para {tumor_name} en {output_path}')

print("Proceso completado.")
