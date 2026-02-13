import pennylane as qml
from pennylane import numpy as np
import numpy as _np
import time

class QuantumDownsampler:
    """
    Clase para manejar la capa de downsampling cuántico (Propuesta 2).
    Inicializa el dispositivo y el circuito una sola vez para eficiencia.
    """
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        # Usamos default.qubit por compatibilidad. 
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
            
        # Usamos interface='numpy' para ejecución rápida sin grafos de computación de Torch/TF
        # ya que esta capa actúa como extractor de características fijo (pre-procesamiento).
        self.qnode = qml.QNode(self._circuit, self.dev, interface="numpy")
        
    def _circuit(self, inputs):
        """
        Circuito cuántico que aplica Amplitude Encoding y capas de entrelazamiento.
        inputs: Array de (Batch_Size, 4) para 2 qubits.
        """
        # 1. Amplitude Encoding
        # normalize=True es crucial. pad_with=0 maneja inputs menores si los hubiera.
        qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_qubits), normalize=True, pad_with=0.)
        
        # 2. Capa "Variacional" (Fija en esta propuesta como extractor)
        phi = np.pi / 2 
        
        # Capa de Rotaciones Controladas
        for i in range(self.n_qubits):
            control = i
            target = (i + 1) % self.n_qubits
            qml.CRZ(phi, wires=[control, target])
            qml.CRX(phi, wires=[control, target])

        # Capa de Entrelazamiento (CZ)
        for i in range(self.n_qubits):
            control = i
            target = (i + 1) % self.n_qubits
            qml.CZ(wires=[control, target])

        # Medición: Valor esperado de PauliZ en qubit 0
        return qml.expval(qml.PauliZ(0))

    def _preprocess_image_blocks(self, image, kernel_size=2, stride=2):
        """
        Divide la imagen en bloques de 2x2 vectorizados.
        """
        h, w = image.shape
        # Recorte para asegurar divisibilidad por stride
        h_new = (h // stride) * stride
        w_new = (w // stride) * stride
        image = image[:h_new, :w_new]
        
        out_h = h_new // stride
        out_w = w_new // stride
        
        # reshape eficiente: (H/2, 2, W/2, 2) -> (H/2, W/2, 2, 2)
        blocks = image.reshape(out_h, stride, out_w, stride)
        blocks = blocks.transpose(0, 2, 1, 3)
        # aplanar bloques: (N_blocks, 4)
        blocks = blocks.reshape(-1, kernel_size * stride)
        
        return blocks, out_h, out_w

    def process_image(self, image, verbose=False):
        """
        Procesa una imagen completa: 
        128x128 -> Bloques 2x2 -> Circuito Cuántico -> 64x64 Feature Map
        """
        start_time = time.time()
        
        # 1. Preprocesar (Sliding Window / Bloques)
        blocks, out_h, out_w = self._preprocess_image_blocks(image)
        
        # 2. Normalización L2 por bloque (Vital para Amplitude Encoding)
        # AmplitudeEmbedding espera vectores unitarios.
        # Calculamos norma por fila
        norms = _np.linalg.norm(blocks, axis=1, keepdims=True)
        
        # Evitar división por cero en bloques negros (todo 0)
        # Asignamos estado base |00> -> [1, 0, 0, 0]
        zero_mask = norms.flatten() < 1e-9
        if _np.any(zero_mask):
            blocks[zero_mask] = [1.0, 0.0, 0.0, 0.0]
            norms[zero_mask] = 1.0
            
        normalized_blocks = blocks / norms
        
        # 3. Ejecución del Circuito (Batch)
        # PennyLane difunde (broadcasts) el input sobre el circuito
        results = self.qnode(normalized_blocks)
        
        # 4. Reconstrucción
        # results es un array 1D de longitud N_blocks
        processed_image = _np.array(results).reshape(out_h, out_w)
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"Imagen procesada en {elapsed:.4f}s")
            
        return processed_image