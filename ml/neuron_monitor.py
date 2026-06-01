from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class LayerStats:
    """Estatísticas de ativação de uma camada."""

    layer_name: str
    mean: float = 0.0
    std: float = 0.0
    dead_neurons_pct: float = 0.0   # % de neurônios com ativação ~0.
    saturated_pct: float = 0.0      # % saturados (> 0.99 em sigmoid/tanh).
    max_activation: float = 0.0
    min_activation: float = 0.0
    total_neurons: int = 0
    update_count: int = 0


class NeuronMonitor:
    """
    Monitor de ativações de redes PyTorch em tempo real.

    Registra hooks em todas as camadas lineares e LSTM do modelo,
    capturando estatísticas por forward pass:
      - % neurônios mortos (ReLU dead neurons: ativação == 0).
      - % neurônios saturados.
      - Mean / Std das ativações por camada.

    Alertas automáticos:
      - Se > 60% neurônios mortos em qualquer camada → alerta.
      - Se gradiente médio < 1e-7 → possível vanishing gradient.
    """

    DEAD_THRESHOLD: float = 0.01
    SATURATION_THRESHOLD: float = 0.99
    DEAD_ALERT_PCT: float = 0.60

    def __init__(self, model: "nn.Module") -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("Instale torch: pip install torch")

        self.model = model
        self._stats: dict[str, LayerStats] = {}
        self._hooks: list[Any] = []
        self._gradient_stats: dict[str, float] = {}
        self._alerts: list[str] = []
        self._attach_hooks()
        self._attach_gradient_hooks()

    # ── Hooks de Forward ──────────────────────────────────────────────────────

    def _attach_hooks(self) -> None:
        """Registra forward hooks em todas as camadas relevantes."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell)):
                hook = module.register_forward_hook(
                    self._make_forward_hook(name)
                )
                self._hooks.append(hook)
                self._stats[name] = LayerStats(layer_name=name)
        logger.debug(
            "NeuronMonitor: hooks registrados.",
            layers=list(self._stats.keys()),
        )

    def _make_forward_hook(self, layer_name: str):
        """Factory de hook por camada."""
        def hook(module, input, output):
            # Para LSTM, output é (output, (h_n, c_n)).
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            if not isinstance(tensor, torch.Tensor):
                return

            with torch.no_grad():
                acts = tensor.detach().float()
                flat = acts.reshape(-1).numpy()

                dead_mask = np.abs(flat) < self.DEAD_THRESHOLD
                saturated_mask = np.abs(flat) > self.SATURATION_THRESHOLD

                stats = self._stats[layer_name]
                stats.mean = float(np.mean(flat))
                stats.std = float(np.std(flat))
                stats.dead_neurons_pct = float(dead_mask.mean())
                stats.saturated_pct = float(saturated_mask.mean())
                stats.max_activation = float(np.max(flat))
                stats.min_activation = float(np.min(flat))
                stats.total_neurons = len(flat)
                stats.update_count += 1

                # Alerta de neurônios mortos.
                if stats.dead_neurons_pct >= self.DEAD_ALERT_PCT:
                    alert = (
                        f"ALERTA: {stats.dead_neurons_pct:.1%} neurônios mortos "
                        f"em '{layer_name}'."
                    )
                    if alert not in self._alerts:
                        self._alerts.append(alert)
                        logger.warning(alert, layer=layer_name)

        return hook

    # ── Hooks de Gradiente ────────────────────────────────────────────────────

    def _attach_gradient_hooks(self) -> None:
        """Registra hooks de gradiente para detectar vanishing/exploding."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                def make_grad_hook(n):
                    def grad_hook(grad):
                        if grad is not None:
                            self._gradient_stats[n] = float(grad.abs().mean().item())
                    return grad_hook
                param.register_hook(make_grad_hook(name))

    # ── API Pública ───────────────────────────────────────────────────────────

    def get_layer_stats(self, layer_name: str) -> LayerStats | None:
        return self._stats.get(layer_name)

    def get_all_stats(self) -> dict[str, LayerStats]:
        return dict(self._stats)

    def get_summary(self) -> dict[str, Any]:
        """
        Retorna resumo para exibição na interface TUI.
        """
        layers_summary = {
            name: {
                "dead_pct": round(s.dead_neurons_pct, 4),
                "saturated_pct": round(s.saturated_pct, 4),
                "mean": round(s.mean, 4),
                "std": round(s.std, 4),
                "neurons": s.total_neurons,
                "updates": s.update_count,
                "status": self._layer_health(s),
            }
            for name, s in self._stats.items()
        }

        vanishing = any(
            v < 1e-7 for v in self._gradient_stats.values()
        )
        exploding = any(
            v > 10.0 for v in self._gradient_stats.values()
        )

        return {
            "layers": layers_summary,
            "alerts": list(self._alerts[-10:]),  # Últimos 10 alertas.
            "gradient_stats": {
                k: round(v, 8) for k, v in self._gradient_stats.items()
            },
            "vanishing_gradient": vanishing,
            "exploding_gradient": exploding,
            "total_layers_monitored": len(self._stats),
        }

    def clear_alerts(self) -> None:
        self._alerts.clear()

    def remove_hooks(self) -> None:
        """Remove todos os hooks — chame ao finalizar treinamento."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("NeuronMonitor: hooks removidos.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _layer_health(self, stats: LayerStats) -> str:
        if stats.dead_neurons_pct >= 0.60:
            return "CRÍTICO"
        if stats.dead_neurons_pct >= 0.30:
            return "ATENÇÃO"
        if stats.saturated_pct >= 0.50:
            return "SATURADO"
        return "OK"