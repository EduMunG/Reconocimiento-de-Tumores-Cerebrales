
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

# --- CONFIGURACIÓN ---
base_dir = 'Preprocesamiento_por_paciente'
tumor_types = ['Meningioma', 'Glioma', 'Adenoma_hipofisario']
output_dir = 'analisis_output'

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# --- ANÁLISIS ---
print("Analizando el número de cortes por paciente...")

# Diccionario para guardar el número de cortes por paciente para cada tipo de tumor
slice_counts = defaultdict(list)

for tumor_type in tumor_types:
    tumor_dir = os.path.join(base_dir, tumor_type)
    if not os.path.isdir(tumor_dir):
        print(f"Directorio no encontrado para {tumor_type}, saltando.")
        continue

    patient_files = [f for f in os.listdir(tumor_dir) if f.endswith('.npz')]

    for patient_file in patient_files:
        try:
            file_path = os.path.join(tumor_dir, patient_file)
            with np.load(file_path) as data:
                # La forma de 'images' es (num_slices, height, width)
                num_slices = data['images'].shape[0]
                slice_counts[tumor_type].append(num_slices)
        except Exception as e:
            print(f"Error procesando el archivo {patient_file}: {e}")

# --- REPORTE Y VISUALIZACIÓN ---
print("\n--- Resultados del Análisis ---")

all_slices = []
for tumor_type in tumor_types:
    counts = slice_counts[tumor_type]
    all_slices.extend(counts)
    if counts:
        print(f"\nTumor: {tumor_type}")
        print(f"  - Número de pacientes: {len(counts)}")
        print(f"  - Mínimo de cortes por paciente: {min(counts)}")
        print(f"  - Máximo de cortes por paciente: {max(counts)}")
        print(f"  - Promedio de cortes por paciente: {statistics.mean(counts):.2f}")
        if len(counts) > 1:
            print(f"  - Desviación estándar: {statistics.stdev(counts):.2f}")
    else:
        print(f"\nTumor: {tumor_type}")
        print("  - No se encontraron datos.")

# --- GENERAR HISTOGRAMA ---
if all_slices:
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 7))
    
    # Crear histogramas para cada tipo de tumor
    for tumor_type in tumor_types:
        plt.hist(slice_counts[tumor_type], bins=range(min(all_slices), max(all_slices) + 1), alpha=0.7, label=tumor_type)

    plt.title('Distribución del Número de Cortes por Paciente', fontsize=16)
    plt.xlabel('Número de Cortes (Imágenes)', fontsize=12)
    plt.ylabel('Número de Pacientes', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Guardar la figura
    output_path = os.path.join(output_dir, 'distribucion_cortes_por_paciente.png')
    plt.savefig(output_path)
    print(f"\nHistograma guardado en: {output_path}")
    print("El histograma muestra cuántos pacientes tienen un cierto número de imágenes.")
else:
    print("\nNo se generó ningún histograma porque no se encontraron datos.")

print("\nAnálisis completado.")
