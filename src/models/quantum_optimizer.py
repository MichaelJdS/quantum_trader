import numpy as np
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    HAS_QML = True
except ImportError:
    HAS_QML = False
    import warnings
    warnings.warn("PennyLane não instalado. Usando otimização clássica fallback.")

def quantum_weight_optimizer(confidences: list[float], regime: str) -> np.ndarray:
    n = len(confidences)
    if n == 0: 
        return np.ones(1)
        
    if not HAS_QML:
        # Fallback robusto usando Softmax para evitar explosão de gradiente
        weights = np.array(confidences) * 1.5
        regime_bias = {"trending": 1.2, "volatile": 0.8, "ranging": 1.0}.get(regime, 1.0)
        weights *= regime_bias
        
        # Cálculo estável de Softmax subtraindo o máximo (evita overflow em exponenciais)
        e_x = np.exp(weights - np.max(weights))
        return e_x / e_x.sum()
    
    dev = qml.device("default.qubit", wires=n)
    regime_map = {"trending": 0.3, "volatile": -0.4, "ranging": 0.0, "neutral": 0.0}
    
    # O PennyLane requer que parâmetros não-treináveis sejam explicitly declarados
    phi = pnp.array(np.ones(n) * regime_map.get(regime, 0.0), requires_grad=False)
    
    @qml.qnode(dev)
    def circuit(weights, phi_params):
        for i in range(n):
            qml.RY(phi_params[i] + weights[i], wires=i)
        for i in range(n - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))
    
    # Parâmetros treináveis declarados corretamente para o otimizador
    weights = pnp.array(np.random.randn(n) * 0.2, requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.05)
    
    for _ in range(10):
        # Apenas 'weights' sofre o step do gradiente
        weights = opt.step(lambda w: circuit(w, phi), weights)
        
    # Extrai os valores numéricos finais e aplica uma Softmax para transformar em probabilidades
    final_weights = np.array(weights)
    e_x = np.exp(final_weights - np.max(final_weights))
    return e_x / e_x.sum()