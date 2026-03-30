"""
visify.core.api
───────────────
The single public entry point: visualize(**kwargs)

Examples
--------
from visify import visualize

# Neural net
visualize(model="neural_network", layers=[4, 6, 6, 2])

# With a real model (PyTorch / Keras duck-typed)
visualize(model=my_pytorch_model, input=x)

# Algorithm
visualize(sort="quick", data=[5, 2, 9, 1, 7, 3])
"""
from __future__ import annotations
from typing import Any

from .registry import Registry

# ── register built-ins on first import ──────────────────────────────────────
def _register_builtins():
    from ..ai.nn import NeuralNetVisualizer
    from ..algorithms.sort import SortVisualizer

    Registry.register("model", NeuralNetVisualizer)
    Registry.register("sort",  SortVisualizer)

_register_builtins()
# ─────────────────────────────────────────────────────────────────────────────


def visualize(**kwargs: Any):
    """
    Dispatch to the right visualizer based on keyword arguments.

    Parameters
    ----------
    model : str | nn.Module
        "neural_network"  — specify architecture via `layers=[...]`
        a PyTorch / Keras model — introspected automatically
    layers : list[int]
        Neuron counts per layer, e.g. [4, 8, 8, 2]
    input : array-like, optional
        Input tensor to run a forward pass and show activations
    sort : str
        "quick" | "merge" | "bubble" | "insertion"
    data : list[int | float]
        Array to sort

    Returns
    -------
    A renderer object. In VS Code / Jupyter notebooks it auto-displays.
    Call .show() to force display, or .html() to get the raw HTML string.
    """
    visualizer_cls, key, value = Registry.resolve(kwargs)
    viz = visualizer_cls(key, value, kwargs)
    return viz.render()
