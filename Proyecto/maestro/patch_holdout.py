import os
import numpy as np
from maestro.loader import *
from maestro.modelos import *
from maestro.utils import *
from maestro.experimento_total import run_experiment, generate_tables, RESULTADOS_DIR

if __name__ == "__main__":
    # 1. Datos previos recuperados del archivo .md
    # Formato: global_results[modelo][escenario] = {metrics}
    model_types = ['replica', 'p1', 'p2', 'p3']
    scenarios = ['unbalanced', 'balanced_loocv', 'balanced_5fold', 'balanced_holdout']
    
    global_results = {m: {s: {} for s in scenarios} for m in model_types}
    
    # Rellenar con los datos que ya tenemos (Tabla 1, 2, 3)
    # Replica
    global_results['replica']['unbalanced'] = {'accuracy': 0.7405, 'f1_macro': 0.7399, 'auc': 0.8840}
    global_results['replica']['balanced_loocv'] = {'accuracy': 0.9528, 'f1_macro': 0.8933, 'auc': 0.0000}
    global_results['replica']['balanced_5fold'] = {'accuracy': 0.7874, 'f1_macro': 0.7756, 'auc': 0.9043}
    
    # P1
    global_results['p1']['unbalanced'] = {'accuracy': 0.8429, 'f1_macro': 0.8386, 'auc': 0.9290}
    global_results['p1']['balanced_loocv'] = {'accuracy': 0.9583, 'f1_macro': 0.9337, 'auc': 0.0000}
    global_results['p1']['balanced_5fold'] = {'accuracy': 0.8828, 'f1_macro': 0.8714, 'auc': 0.9517}
    
    # P2
    global_results['p2']['unbalanced'] = {'accuracy': 0.8333, 'f1_macro': 0.8303, 'auc': 0.9360}
    global_results['p2']['balanced_loocv'] = {'accuracy': 0.9595, 'f1_macro': 0.9272, 'auc': 0.0000}
    global_results['p2']['balanced_5fold'] = {'accuracy': 0.8656, 'f1_macro': 0.8578, 'auc': 0.9411}
    
    # P3
    global_results['p3']['unbalanced'] = {'accuracy': 0.7714, 'f1_macro': 0.7725, 'auc': 0.8912}
    global_results['p3']['balanced_loocv'] = {'accuracy': 0.8788, 'f1_macro': 0.7436, 'auc': 0.0000}
    global_results['p3']['balanced_5fold'] = {'accuracy': 0.7741, 'f1_macro': 0.7669, 'auc': 0.9012}

    # 2. Ejecutar SOLO el nuevo escenario: balanced_holdout
    params = {
        'lr': 0.001,
        'epochs': 50
    }
    
    print(">>> Ejecutando SOLO el escenario: balanced_holdout")
    for m in model_types:
        metrics = run_experiment(m, 'balanced_holdout', params)
        global_results[m]['balanced_holdout'] = metrics
            
    # 3. Generar y guardar reporte completo (incluyendo el parche)
    report = generate_tables(global_results)
    print("\n" + report)
    
    output_path = os.path.join(RESULTADOS_DIR, "tablas_finales_completas.md")
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"\n¡Éxito! Resultados parchados y guardados en {output_path}")
