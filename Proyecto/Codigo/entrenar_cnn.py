import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# --- 1. Carga y Preparación de Datos ---

def cargar_datos_preprocesados(base_dir):
    """Carga los datos desde las carpetas de tumores preprocesados."""
    imagenes = []
    etiquetas = []
    tumor_types = ['Meningioma', 'Glioma', 'Adenoma_hipofisario']

    print(f"Cargando datos desde {base_dir}...")
    for tumor_name in tumor_types:
        file_path = os.path.join(base_dir, tumor_name, f'{tumor_name}_preprocessed.npz')
        if not os.path.exists(file_path):
            print(f"Advertencia: No se encontró el archivo {file_path}")
            continue
        
        with np.load(file_path) as data:
            imagenes.append(data['images'])
            etiquetas.append(data['labels'])
            print(f"- Cargadas {len(data['images'])} imágenes de {tumor_name}")

    if not imagenes:
        print("Error: No se cargaron datos. Asegúrate de que los archivos .npz existen.")
        return None, None

    # Combinar los datos de todas las carpetas
    X = np.concatenate(imagenes, axis=0)
    y = np.concatenate(etiquetas, axis=0)
    
    return X, y

# Cargar los datos
base_data_dir = 'preprocessed_data_por_tumor'
X, y = cargar_datos_preprocesados(base_data_dir)

if X is not None:
    # Las etiquetas están como 1, 2, 3. Las convertimos a 0, 1, 2 para Keras.
    y = y - 1

    # Añadir la dimensión del canal (escala de grises, 1 canal)
    X = np.expand_dims(X, axis=-1)

    # Dividir en conjuntos de entrenamiento y prueba (80% train, 20% test)
    # Usamos stratify para mantener la proporción de clases en ambos conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convertir etiquetas a formato one-hot (ej: 2 -> [0, 0, 1])
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    print("\nDatos listos para el entrenamiento:")
    print(f"Forma de X_train: {X_train.shape}")
    print(f"Forma de y_train (one-hot): {y_train_cat.shape}")
    print(f"Forma de X_test: {X_test.shape}")
    print(f"Forma de y_test (one-hot): {y_test_cat.shape}")

    # --- 2. Creación del Modelo CNN ---

    def crear_modelo_cnn(input_shape, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Crear el modelo
    input_shape = X_train.shape[1:]
    num_classes = 3
    modelo_cnn = crear_modelo_cnn(input_shape, num_classes)
    modelo_cnn.summary()

    # --- 3. Entrenamiento del Modelo ---

    print("\nIniciando entrenamiento...")
    history = modelo_cnn.fit(
        X_train, y_train_cat,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test_cat)
    )
    print("Entrenamiento completado.")

    # --- 4. Evaluación y Visualización ---

    print("\nEvaluando el modelo...")
    loss, accuracy = modelo_cnn.evaluate(X_test, y_test_cat, verbose=0)
    print(f'Pérdida en el conjunto de prueba: {loss:.4f}')
    print(f'Precisión en el conjunto de prueba: {accuracy:.4f}')

    # Graficar historial de entrenamiento
    plt.figure(figsize=(12, 5))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.tight_layout()
    plt.show()
