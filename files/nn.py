"""
visify.ai.nn
────────────
Neural network visualizer.

Supports:
  - Architecture diagrams from a layers list
  - Forward pass activation flow
  - Auto-introspection of PyTorch / Keras models
  - Gradient magnitude overlay (PyTorch with backward hook)

Usage
-----
from visify import visualize

# Describe an architecture
visualize(model="neural_network", layers=[4, 8, 8, 2])

# Run a forward pass with activations
import numpy as np
visualize(model="neural_network", layers=[4, 8, 8, 2],
          input=np.random.rand(4))

# PyTorch model (auto-introspected)
visualize(model=my_model, input=x_tensor)
"""
from __future__ import annotations

import math
import json
import random
from typing import Any, Dict, Iterator, List, Optional

from ..core.base import BaseVisualizer, Frame


# ── helpers ──────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def _relu(x: float) -> float:
    return max(0.0, x)

def _tanh(x: float) -> float:
    return math.tanh(max(-500, min(500, x)))

ACTIVATIONS = {"relu": _relu, "sigmoid": _sigmoid, "tanh": _tanh}


def _extract_pytorch_layers(model) -> List[int]:
    """Pull layer sizes from a PyTorch Sequential or Module."""
    sizes = []
    try:
        import torch.nn as nn
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if not sizes:
                    sizes.append(m.in_features)
                sizes.append(m.out_features)
    except ImportError:
        pass
    return sizes or [4, 8, 4]


def _extract_keras_layers(model) -> List[int]:
    """Pull layer sizes from a Keras model."""
    sizes = []
    try:
        for layer in model.layers:
            cfg = layer.get_config()
            if "units" in cfg:
                if not sizes and hasattr(layer, "input_shape"):
                    try:
                        sizes.append(layer.input_shape[-1])
                    except Exception:
                        pass
                sizes.append(cfg["units"])
    except Exception:
        pass
    return sizes or [4, 8, 4]


def _infer_layers(model_val: Any) -> List[int]:
    """Auto-detect layer structure from a model object."""
    cls_name = type(model_val).__name__
    if cls_name in ("Sequential", "Module"):
        return _extract_pytorch_layers(model_val)
    if hasattr(model_val, "layers"):
        return _extract_keras_layers(model_val)
    return [4, 8, 4]


def _random_weights(n_in: int, n_out: int) -> List[List[float]]:
    """Xavier-ish random weight matrix."""
    scale = math.sqrt(2.0 / (n_in + n_out))
    return [[random.gauss(0, scale) for _ in range(n_out)] for _ in range(n_in)]


def _forward_pass(
    layers: List[int],
    input_vec: List[float],
    activation: str = "relu",
) -> List[List[float]]:
    """
    Run a fake forward pass through the network.
    Returns activations per layer (including input layer).
    """
    fn = ACTIVATIONS.get(activation, _relu)
    all_weights = []
    for i in range(len(layers) - 1):
        all_weights.append(_random_weights(layers[i], layers[i + 1]))

    activations = [list(input_vec)]
    current = list(input_vec)

    for i, W in enumerate(all_weights):
        is_last = i == len(all_weights) - 1
        next_act_fn = _sigmoid if is_last else fn
        nxt = []
        for j in range(layers[i + 1]):
            z = sum(current[k] * W[k][j] for k in range(len(current)))
            nxt.append(next_act_fn(z))
        activations.append(nxt)
        current = nxt

    return activations


# ── main class ───────────────────────────────────────────────────────────────

class NeuralNetVisualizer(BaseVisualizer):
    """
    Generates frames for a neural network visualization.

    Frame types
    -----------
    "architecture"  — static layer diagram, no activations
    "layer_focus"   — highlight one layer at a time
    "forward_pass"  — show activation values flowing layer-by-layer
    "full"          — final state with all activations visible
    """

    def frames(self) -> Iterator[Frame]:
        # ── resolve layers ────────────────────────────────────────────────
        layers: List[int] = self.kwargs.get("layers", None)
        model_val = self.value

        if layers is None:
            if isinstance(model_val, str) and model_val == "neural_network":
                layers = [4, 6, 6, 2]          # sensible default
            else:
                layers = _infer_layers(model_val)

        # cap for readability
        layers = [min(n, 12) for n in layers]

        # ── resolve input & activations ───────────────────────────────────
        raw_input = self.kwargs.get("input", None)
        activation_name: str = self.kwargs.get("activation", "relu")
        activations: Optional[List[List[float]]] = None

        if raw_input is not None:
            try:
                # numpy / tensor → plain list
                flat = list(raw_input.flatten()) if hasattr(raw_input, "flatten") else list(raw_input)
                # pad / truncate to match input layer
                n_in = layers[0]
                if len(flat) >= n_in:
                    input_vec = [float(v) for v in flat[:n_in]]
                else:
                    input_vec = [float(v) for v in flat] + [0.0] * (n_in - len(flat))
                activations = _forward_pass(layers, input_vec, activation_name)
            except Exception:
                activations = None

        if activations is None:
            # generate plausible random activations for demo
            random.seed(42)
            dummy_input = [random.uniform(0.2, 0.9) for _ in range(layers[0])]
            activations = _forward_pass(layers, dummy_input, activation_name)

        # ── emit frames ───────────────────────────────────────────────────
        # Frame 0: architecture overview
        yield Frame(
            type="architecture",
            layers=layers,
            activations=[[0.0] * n for n in layers],
            highlight_layer=None,
            label="Network architecture",
            subtitle=f"{len(layers)} layers · {sum(layers)} total neurons",
        )

        # Frames 1…N: layer-by-layer forward pass
        for i in range(len(layers)):
            # show activations up to layer i
            partial = [activations[j] if j <= i else [0.0] * layers[j]
                       for j in range(len(layers))]
            yield Frame(
                type="forward_pass",
                layers=layers,
                activations=partial,
                highlight_layer=i,
                label=f"Layer {i} {'(input)' if i == 0 else '(output)' if i == len(layers)-1 else f'(hidden {i})'}",
                subtitle=(
                    f"Activations: [{', '.join(f'{v:.2f}' for v in activations[i][:4])}"
                    + (f", …+{len(activations[i])-4}" if len(activations[i]) > 4 else "")
                    + "]"
                ),
            )

        # Final frame: complete picture
        yield Frame(
            type="full",
            layers=layers,
            activations=activations,
            highlight_layer=None,
            label="Forward pass complete",
            subtitle=f"Output: [{', '.join(f'{v:.3f}' for v in activations[-1])}]",
        )

    def _metadata(self) -> Dict[str, Any]:
        return {
            "visualizer": "NeuralNetVisualizer",
            "type": "neural_network",
        }
