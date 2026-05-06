import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class BrainTumorDataset(Dataset):
    """
    Dataset para imágenes de tumores cerebrales.
    Soporta imágenes (N, H, W) y etiquetas (N,).
    """
    def __init__(self, images, labels, transform=None, normalize=True):
        """
        Args:
            images (np.ndarray): Array de imágenes (N, H, W).
            labels (np.ndarray): Array de etiquetas (N,).
            transform (callable, optional): Transformaciones opcionales (ej. torchvision.transforms).
            normalize (bool): Si es True, normaliza las imágenes a [0, 1].
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Asegurar que sea float32 para el procesamiento y tensor
        image = self.images[idx].astype(np.float32)
        label = self.labels[idx]
        
        # Normalización básica: escalar de [min, max] a [0, 1]
        if self.normalize:
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                image = (image - min_val) / (max_val - min_val)
            else:
                image = np.zeros_like(image)
        
        # Añadir dimensión de canal (1, H, W) para PyTorch
        if image.ndim == 2:
            image = image[np.newaxis, :, :]
            
        # Convertir a Tensores de PyTorch
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.tensor(label).long()
        
        # Aplicar transformaciones adicionales si se proveen
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor

def decodificar_id_paciente(datos_id):
    """
    Decodifica el ID del paciente desde el formato almacenado en archivos .mat (h5py).
    """
    try:
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

def load_unbalanced_data(mat_dir):
    """
    Lee archivos .mat desde el directorio especificado.
    Extrae la imagen, etiqueta y el ID del paciente para cada archivo.
    
    Args:
        mat_dir (str): Directorio con archivos .mat (DataSet/Tumores/).
        
    Returns:
        X (np.ndarray): Array de imágenes (N, 512, 512).
        y (np.ndarray): Array de etiquetas (N,) mapeadas a [0, 1, 2].
        groups (np.ndarray): Array de IDs de pacientes para validación por grupos.
    """
    X = []
    y = []
    groups = []
    
    # Mapeo de etiquetas original -> 0-indexed:
    # 1: Meningioma -> 0
    # 2: Glioma -> 1
    # 3: Pituitary -> 2
    label_map = {1: 0, 2: 1, 3: 2}
    
    if not os.path.exists(mat_dir):
        # Si no existe, no podemos continuar
        pass
        
    if not os.path.isdir(mat_dir):
        raise ValueError(f"La ruta {mat_dir} no es un directorio válido.")

    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    # Ordenar por nombre numérico si es posible
    mat_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    for filename in tqdm(mat_files, desc="Cargando .mat (desbalanceado)"):
        filepath = os.path.join(mat_dir, filename)
        try:
            with h5py.File(filepath, 'r') as f:
                cjdata = f['cjdata']
                # Extraer imagen y etiqueta
                image = np.array(cjdata['image'])
                label_orig = int(np.array(cjdata['label']).flatten()[0])
                
                # Mapear etiqueta
                label = label_map.get(label_orig, label_orig - 1)
                
                # Extraer PID
                pid = decodificar_id_paciente(cjdata['PID'])
                
                X.append(image)
                y.append(label)
                groups.append(pid)
        except Exception:
            continue
            
    return np.array(X), np.array(y), np.array(groups)

def load_balanced_data(npz_dir):
    """
    Lee archivos .npz desde el directorio especificado de forma recursiva.
    
    Args:
        npz_dir (str): Directorio raíz del dataset balanceado (Codigo/Dataset_Balanceado_700/).
        
    Returns:
        X (np.ndarray): Array de imágenes.
        y (np.ndarray): Array de etiquetas (N,) mapeadas a [0, 1, 2].
        groups (np.ndarray): Array de IDs de pacientes.
    """
    X = []
    y = []
    groups = []
    
    # Mapeo de etiquetas
    label_map = {1: 0, 2: 1, 3: 2}
    
    if not os.path.exists(npz_dir):
        raise FileNotFoundError(f"Directorio balanceado no encontrado: {npz_dir}")
    
    for root, _, files in os.walk(npz_dir):
        npz_files = [f for f in files if f.endswith('.npz')]
        npz_files.sort()
        
        for filename in npz_files:
            filepath = os.path.join(root, filename)
            try:
                data = np.load(filepath)
                imgs = data['imagenes'] if 'imagenes' in data else data['images']
                label_orig = int(data['etiqueta']) if 'etiqueta' in data else int(data['label'])
                label = label_map.get(label_orig, label_orig - 1)
                
                pid = filename.replace('patient_', '').replace('paciente_', '').replace('.npz', '')
                
                if imgs.ndim == 3: # (N, H, W)
                    X.extend(imgs)
                    y.extend([label] * len(imgs))
                    groups.extend([pid] * len(imgs))
                elif imgs.ndim == 2: # (H, W)
                    X.append(imgs)
                    y.append(label)
                    groups.append(pid)
            except Exception:
                continue
                    
    return np.array(X), np.array(y), np.array(groups)
