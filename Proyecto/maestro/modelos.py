import os
import hashlib
import numpy as _np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Ensure cache directory exists
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class RedNeuronalGeneral(nn.Module):
    def __init__(self, input_shape, num_classes=3):
        super(RedNeuronalGeneral, self).__init__()
        # Dynamic calculation of flattened input dimension
        self.input_dim = _np.prod(input_shape)
        
        self.flatten = nn.Flatten()
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits

class QuantumProcessor:
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="numpy")

    def _circuit(self, inputs):
        # 1. Amplitude Encoding
        qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_qubits), normalize=True, pad_with=0.)
        
        # 2. Fixed Rotations Layer
        phi = np.pi / 2 
        for i in range(self.n_qubits):
            control = i
            target = (i + 1) % self.n_qubits
            qml.CRZ(phi, wires=[control, target])
            qml.CRX(phi, wires=[control, target])

        # 3. Entanglement Layer
        for i in range(self.n_qubits):
            control = i
            target = (i + 1) % self.n_qubits
            qml.CZ(wires=[control, target])

        return qml.expval(qml.PauliZ(0))

    def _preprocess_image_blocks(self, image, kernel_size=2, stride=2):
        # Use sliding_window_view for flexible stride and kernel size
        windows = _np.lib.stride_tricks.sliding_window_view(image, (kernel_size, kernel_size))
        
        # Select windows according to stride
        strided_windows = windows[::stride, ::stride]
        
        out_h, out_w, kh, kw = strided_windows.shape
        # Flatten each window into a vector
        blocks = strided_windows.reshape(-1, kh * kw)
        
        return blocks, out_h, out_w

    def _apply_quantum_conv(self, image, kernel_size=2, stride=2):
        blocks, out_h, out_w = self._preprocess_image_blocks(image, kernel_size, stride)
        
        # L2 Normalization per block (essential for Amplitude Embedding)
        block_size = blocks.shape[1]
        norms = _np.linalg.norm(blocks, axis=1, keepdims=True)
        
        # Handle zero-norm blocks by replacing them with a unit vector of the correct size
        zero_mask = norms.flatten() < 1e-9
        if _np.any(zero_mask):
            unit_vec = _np.zeros(block_size)
            unit_vec[0] = 1.0
            blocks[zero_mask] = unit_vec
            norms[zero_mask] = 1.0
            
        normalized_blocks = blocks / norms
        
        results = self.qnode(normalized_blocks)
        return _np.array(results).reshape(out_h, out_w)

    def process(self, image, proposal_id, cache_dir):
        # Disk caching logic
        img_hash = hashlib.sha256(image.tobytes() + str(proposal_id).encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{img_hash}.npy")
        
        if os.path.exists(cache_path):
            return _np.load(cache_path)
        
        # Processing logic per proposal
        if proposal_id == 1:
            # P1: 128x128 output (stride=1 with edge padding)
            padded_img = _np.pad(image, ((0, 1), (0, 1)), mode='edge')
            result = self._apply_quantum_conv(padded_img, kernel_size=2, stride=1)
        elif proposal_id == 2:
            # P2: 64x64 output (single layer stride=2)
            result = self._apply_quantum_conv(image, kernel_size=2, stride=2)
        elif proposal_id == 3:
            # P3: 32x32 output (two layers of stride=2)
            res_l1 = self._apply_quantum_conv(image, kernel_size=2, stride=2)
            result = self._apply_quantum_conv(res_l1, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown proposal_id: {proposal_id}")
            
        # Save to disk
        os.makedirs(cache_dir, exist_ok=True)
        _np.save(cache_path, result)
        
        return result
