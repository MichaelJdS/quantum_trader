import numpy as np
try:
    import pennylane as qml
    HAS_QML = True
except ImportError:
    HAS_QML = False
    import warnings
    warnings.warn("PennyLane não instalado. Usando otimização clássica fallback.")

def quantum_weight_optimizer(confidences: list[float], regime: str) -> np.ndarray:
    n = len(confidences)
    if n == 0: return np.ones(1) / n
    
    if not HAS_QML:
        weights = np.array(confidences) * 1.5
        regime_bias = {"trending": 1.2, "volatile": 0.8, "ranging": 1.0}.get(regime, 1.0)
        weights *= regime_bias
        return weights / weights.sum()
    
    dev = qml.device("default.qubit", wires=n)
    regime_map = {"trending": 0.3, "volatile": -0.4, "ranging": 0.0, "neutral": 0.0}
    phi = np.ones(n) * regime_map.get(regime, 0.0)
    
    @qml.qnode(dev)
    def circuit(weights, phi):
        for i in range(n):
            qml.RY(phi[i] + weights[i], wires=i)
        qml.Barrier()
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))
    
    weights = np.random.randn(n) * 0.2
    opt = qml.GradientDescentOptimizer(stepsize=0.05)
    for _ in range(10):
        weights = opt.step(circuit, weights, phi)
        
    probs = np.exp(weights) / np.sum(np.exp(weights))
    return probs