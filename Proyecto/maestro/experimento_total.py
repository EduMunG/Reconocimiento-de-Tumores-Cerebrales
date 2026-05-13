import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from tqdm import tqdm

# Asegurar que el directorio raíz del proyecto esté en el path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image

# Importar módulos de maestro de forma robusta
try:
    from maestro.loader import BrainTumorDataset, load_unbalanced_data, load_balanced_data
    from maestro.modelos import RedNeuronalGeneral, QuantumProcessor
    from maestro.utils import calculate_metrics, save_confusion_matrix, Profiler
except ImportError:
    from loader import BrainTumorDataset, load_unbalanced_data, load_balanced_data
    from modelos import RedNeuronalGeneral, QuantumProcessor
    from utils import calculate_metrics, save_confusion_matrix, Profiler

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT_DIR = os.path.join(BASE_DIR, "DataSet", "Tumores")
NPZ_DIR = os.path.join(BASE_DIR, "Codigo", "Dataset_Balanceado_700")
RESULTADOS_DIR = os.path.join(os.path.dirname(__file__), "resultados")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Detectar dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Usando dispositivo: {device}")

def resize_image(img, size):
    if cv2:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    else:
        # Fallback a PIL
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize(size, resample=Image.Resampling.LANCZOS)
        return np.array(img_pil)

def dataset_analysis(X, y, groups):
    """
    Realiza el análisis 'Antes': estadísticas del dataset.
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    n_patients = len(np.unique(groups))
    print("\n--- ANÁLISIS DEL DATASET (Antes) ---")
    print(f"Total de imágenes: {len(X)}")
    print(f"Total de pacientes: {n_patients}")
    for label, count in zip(unique_labels, counts):
        print(f"Clase {label}: {count} imágenes")
    return {"total_images": len(X), "total_patients": n_patients}

def preprocess_dataset(X, model_type, qp):
    """
    Preprocesa el dataset según el modelo.
    """
    X_processed = []
    
    # Definir tamaño objetivo inicial
    # Para replica usamos 64x64 como en el notebook original
    # Para P1, P2, P3 usamos 128x128 para que las capas cuánticas den el tamaño esperado
    target_size = 64 if model_type == 'replica' else 128
    
    desc = f"Preprocesando {model_type}"
    for i in tqdm(range(len(X)), desc=desc):
        img = X[i]
        # Redimensionar
        img_resized = resize_image(img, (target_size, target_size))
        
        if model_type == 'replica':
            X_processed.append(img_resized)
        else:
            # Procesamiento cuántico
            proposal_id = int(model_type[1]) # p1 -> 1, p2 -> 2, p3 -> 3
            q_img = qp.process(img_resized, proposal_id, CACHE_DIR)
            X_processed.append(q_img)
            
    return np.array(X_processed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
    return running_loss / len(loader), calculate_metrics(all_labels, all_preds, all_probs)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            
    return running_loss / len(loader), calculate_metrics(all_labels, all_preds, all_probs)

def run_experiment(model_type, data_scenario, params):
    print(f"\n>>> Ejecutando Experimento: Modelo={model_type}, Escenario={data_scenario}")
    
    # 1. Cargar Datos
    if 'balanced' in data_scenario:
        X, y, groups = load_balanced_data(NPZ_DIR)
    else:
        X, y, groups = load_unbalanced_data(MAT_DIR)
        
    dataset_stats = dataset_analysis(X, y, groups)
    
    # 2. Preprocesamiento (incluyendo Quantum si aplica)
    # Medir recursos durante preprocesamiento (Durante - Fase 1)
    prep_profiler = Profiler()
    prep_profiler.start()
    
    qp = QuantumProcessor(n_qubits=2)
    X_p = preprocess_dataset(X, model_type, qp)
    
    prep_time = prep_profiler.get_elapsed()
    prep_ram, prep_gpu = prep_profiler.get_memory_usage()

    # 3. Definir Estrategia de Validación
    if data_scenario == 'unbalanced':
        gkf = GroupKFold(n_splits=5)
        train_idx, test_idx = next(gkf.split(X_p, y, groups))
        splits = [(train_idx, test_idx)]
    elif data_scenario == 'balanced_5fold':
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X_p, y, groups))
    elif data_scenario == 'balanced_loocv':
        logo = LeaveOneGroupOut()
        splits = list(logo.split(X_p, y, groups))
    elif data_scenario == 'balanced_holdout':
        from sklearn.model_selection import GroupShuffleSplit
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_val_idx, test_idx = next(gss_test.split(X_p, y, groups))
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
        X_tmp, y_tmp, groups_tmp = X_p[train_val_idx], y[train_val_idx], groups[train_val_idx]
        train_idx_sub, val_idx_sub = next(gss_val.split(X_tmp, y_tmp, groups_tmp))
        train_idx = train_val_idx[train_idx_sub]
        val_idx = train_val_idx[val_idx_sub]
        splits = [(train_idx, val_idx, test_idx)]
    
    all_fold_metrics = []
    
    # Medir recursos durante entrenamiento (Durante - Fase 2)
    train_profiler = Profiler()
    train_profiler.start()

    for fold, split_data in enumerate(splits):
        if len(split_data) == 3:
            t_idx, v_idx, test_idx = split_data
            test_ds = BrainTumorDataset(X_p[test_idx], y[test_idx])
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        else:
            t_idx, v_idx = split_data
            test_loader = None

        if len(splits) > 1:
            print(f"  Fold {fold+1}/{len(splits)}")
            
        X_train, X_val = X_p[t_idx], X_p[v_idx]
        y_train, y_val = y[t_idx], y[v_idx]
        
        train_ds = BrainTumorDataset(X_train, y_train)
        val_ds = BrainTumorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Inicializar Modelo
        input_shape = X_train[0].shape
        if len(input_shape) == 2:
            input_shape = (1, input_shape[0], input_shape[1])
            
        model = RedNeuronalGeneral(input_shape=input_shape, num_classes=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = -1.0
        fold_best_metrics = None
        
        for epoch in range(params['epochs']):
            t_loss, t_met = train_one_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_met = validate(model, val_loader, criterion, device)
            
            if v_met['accuracy'] >= best_val_acc:
                best_val_acc = v_met['accuracy']
                if test_loader:
                    _, test_met = validate(model, test_loader, criterion, device)
                    fold_best_metrics = test_met
                else:
                    fold_best_metrics = v_met
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{params['epochs']} - Loss: {t_loss:.4f}, Acc: {t_met['accuracy']:.4f} | Val Acc: {v_met['accuracy']:.4f}")
        
        all_fold_metrics.append(fold_best_metrics)

    train_time = train_profiler.get_elapsed()
    train_ram, train_gpu = train_profiler.get_memory_usage()

    avg_metrics = {
        'val_accuracy': best_val_acc,
        'accuracy': np.mean([m['accuracy'] for m in all_fold_metrics]),
        'f1_macro': np.mean([m['f1_macro'] for m in all_fold_metrics]),
        'auc': np.mean([m['auc'] for m in all_fold_metrics]),
        'resources': {
            'prep_time_s': prep_time,
            'train_time_s': train_time,
            'ram_gb': max(prep_ram, train_ram),
            'gpu_mb': max(prep_gpu, train_gpu)
        }
    }

    return avg_metrics

def generate_tables(results, scenarios):
    model_types = ['replica', 'p1', 'p2', 'p3']
    output = "# Resultados Finales de Experimentos (Análisis Estadístico)\n\n"
    
    titles = {
        'unbalanced': "Dataset Desbalanceado (Hold-out 70/15/15)",
        'balanced_5fold': "Dataset Balanceado (5-Fold CV)",
        'balanced_holdout': "Dataset Balanceado (LOSO Hold-out 70/15/15)"
    }
    
    for i, scenario in enumerate(scenarios):
        title = titles.get(scenario, f"Escenario {scenario}")
        output += f"### {title}\n\n"
        
        output += "#### Rendimiento de Clasificación (Después)\n"
        output += "| Modelo | Val. Acc | Test Acc | F1-Score | AUC |\n"
        output += "| --- | --- | --- | --- | --- |\n"
        for m_type in model_types:
            m = results[m_type][scenario]
            output += f"| {m_type.upper()} | {m['val_accuracy']:.4f} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {m['auc']:.4f} |\n"
        output += "\n"
        
        output += "#### Eficiencia de Recursos (Durante)\n"
        output += "| Modelo | Tiempo Prep (s) | Tiempo Train (s) | RAM (GB) | GPU (MB) |\n"
        output += "| --- | --- | --- | --- | --- |\n"
        for m_type in model_types:
            res = results[m_type][scenario]['resources']
            output += f"| {m_type.upper()} | {res['prep_time_s']:.2f} | {res['train_time_s']:.2f} | {res['ram_gb']:.2f} | {res['gpu_mb']:.2f} |\n"
        output += "\n"
        
    return output

if __name__ == "__main__":
    model_types = ['replica', 'p1', 'p2', 'p3']
    scenarios = ['balanced_5fold', 'balanced_holdout']
    
    params = {
        'lr': 0.001,
        'epochs': 50
    }
    
    global_results = {m: {s: {} for s in scenarios} for m in model_types}
    
    for s in scenarios:
        for m in model_types:
            metrics = run_experiment(m, s, params)
            global_results[m][s] = metrics
            
    report = generate_tables(global_results, scenarios)
    print("\n" + report)
    
    with open(os.path.join(RESULTADOS_DIR, "tablas_finales.md"), "w") as f:
        f.write(report)
    
    print(f"Resultados guardados en {os.path.join(RESULTADOS_DIR, 'tablas_finales.md')}")
