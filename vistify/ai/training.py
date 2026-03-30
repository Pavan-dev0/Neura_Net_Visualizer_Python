"""
visify.ai.training
──────────────────
A from-scratch neural network built for students.
No PyTorch. No Keras. Just NumPy + math you can actually read.

Usage
-----
from visify.ai.training import StudentNet

# XOR problem
net = StudentNet(layers=[2, 4, 1], activation="sigmoid")
net.train(
    X=[[0,0],[0,1],[1,0],[1,1]],
    y=[[0],  [1],  [1],  [0]],
    epochs=500,
    lr=0.5
)
net.visualize()   # opens the full interactive visualization
"""
from __future__ import annotations
from importlib import metadata
import math
import os
from pydoc import html
import random
import sys
from typing import List, Tuple, Optional

# ── pure-python math (no numpy required) ────────────────────────────────────

def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def _sigmoid_deriv(out: float) -> float:
    return out * (1.0 - out)

def _relu(x: float) -> float:
    return max(0.0, x)

def _relu_deriv(out: float) -> float:
    return 1.0 if out > 0 else 0.0

def _tanh(x: float) -> float:
    return math.tanh(max(-500.0, min(500.0, x)))

def _tanh_deriv(out: float) -> float:
    return 1.0 - out ** 2

ACTIVATIONS = {
    "sigmoid": (_sigmoid, _sigmoid_deriv),
    "relu":    (_relu,    _relu_deriv),
    "tanh":    (_tanh,    _tanh_deriv),
}

def _mse(pred: List[float], target: List[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)

def _xavier(n_in: int, n_out: int) -> float:
    """Xavier initialisation — keeps gradients stable."""
    limit = math.sqrt(6.0 / (n_in + n_out))
    return random.uniform(-limit, limit)

# ── matrix helpers (plain lists-of-lists) ────────────────────────────────────

Matrix = List[List[float]]
Vector = List[float]

def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]

def _mat_vec(W: Matrix, v: Vector) -> Vector:
    """W @ v"""
    return [sum(W[i][j] * v[j] for j in range(len(v))) for i in range(len(W))]


# ── the network ───────────────────────────────────────────────────────────────

class StudentNet:
    """
    A fully-connected neural network written in pure Python.
    Every line is readable. Every concept is explicit.

    Parameters
    ----------
    layers : list[int]
        Neuron counts per layer including input and output.
        e.g. [2, 4, 4, 1] means 2 inputs, two hidden layers of 4, 1 output.
    activation : "sigmoid" | "relu" | "tanh"
        Activation function for hidden layers. Output always uses sigmoid.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        layers: List[int],
        activation: str = "sigmoid",
        seed: int = 42,
    ):
        random.seed(seed)
        self.layers = layers
        self.activation_name = activation
        self._act, self._act_d = ACTIVATIONS.get(activation, ACTIVATIONS["sigmoid"])

        # weights[i] is the weight matrix between layer i and layer i+1
        # weights[i][j][k] = weight from neuron j in layer i to neuron k in layer i+1
        self.weights: List[Matrix] = []
        self.biases:  List[Vector] = []

        for i in range(len(layers) - 1):
            n_in, n_out = layers[i], layers[i + 1]
            self.weights.append([[_xavier(n_in, n_out) for _ in range(n_out)]
                                  for _ in range(n_in)])
            self.biases.append([0.0] * n_out)

        # training history — stored for visualization
        self.history: List[dict] = []   # one entry per epoch
        self._is_trained = False

    # ── forward pass ─────────────────────────────────────────────────────────

    def _forward(self, x: Vector) -> Tuple[List[Vector], List[Vector]]:
        """
        Full forward pass.
        Returns (pre_activations, activations) for every layer.
        activations[0] is the input, activations[-1] is the output.
        """
        pre_acts  = []       # z values (before activation)
        acts      = [list(x)]  # a values (after activation)

        current = list(x)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = [sum(current[j] * W[j][k] for j in range(len(current))) + b[k]
                 for k in range(len(b))]
            pre_acts.append(z)

            is_last = i == len(self.weights) - 1
            if is_last:
                a = [_sigmoid(zi) for zi in z]   # output always sigmoid
            else:
                a = [self._act(zi) for zi in z]

            acts.append(a)
            current = a

        return pre_acts, acts

    def predict(self, x: Vector) -> Vector:
        _, acts = self._forward(x)
        return acts[-1]

    # ── backward pass (backpropagation) ──────────────────────────────────────

    def _backward(
        self,
        pre_acts: List[Vector],
        acts:     List[Vector],
        target:   Vector,
        lr:       float,
    ) -> float:
        """
        Backpropagation — compute gradients and update weights.
        Returns the loss for this sample.
        """
        n_layers = len(self.layers)
        loss = _mse(acts[-1], target)

        # output layer delta: dL/dz = (pred - target) * sigmoid'(pred)
        deltas = [None] * (n_layers - 1)
        out = acts[-1]
        deltas[-1] = [(out[k] - target[k]) * _sigmoid_deriv(out[k])
                      for k in range(len(out))]

        # hidden layer deltas: propagate backwards
        for i in range(n_layers - 3, -1, -1):
            W_next = self.weights[i + 1]   # (n_i+1 × n_i+2)
            d_next = deltas[i + 1]
            a_i    = acts[i + 1]
            deltas[i] = [
                self._act_d(a_i[j]) * sum(W_next[j][k] * d_next[k]
                                          for k in range(len(d_next)))
                for j in range(len(a_i))
            ]

        # update weights and biases
        for i in range(len(self.weights)):
            a_in = acts[i]
            d    = deltas[i]
            for j in range(len(a_in)):
                for k in range(len(d)):
                    self.weights[i][j][k] -= lr * d[k] * a_in[j]
            for k in range(len(d)):
                self.biases[i][k] -= lr * d[k]

        return loss

    # ── training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        X:             List[Vector],
        y:             List[Vector],
        epochs:        int  = 1000,
        lr:            float = 0.1,
        snapshot_every: int  = 1,
        verbose:       bool  = True,
    ) -> "StudentNet":
        """
        Train the network and record snapshots for visualization.

        Parameters
        ----------
        X : list of input vectors
        y : list of target vectors
        epochs : number of training epochs
        lr : learning rate
        snapshot_every : record a visualization frame every N epochs
        verbose : print loss every 10% of training
        """
        self.history = []
        n = len(X)

        for epoch in range(epochs):
            # shuffle training data each epoch
            indices = list(range(n))
            random.shuffle(indices)

            epoch_loss = 0.0
            all_acts   = []   # activations for each sample this epoch

            for idx in indices:
                pre_acts, acts = self._forward(X[idx])
                loss = self._backward(pre_acts, acts, y[idx], lr)
                epoch_loss += loss
                all_acts.append(acts)

            epoch_loss /= n

            # compute accuracy (for binary classification)
            correct = sum(
                1 for i in range(n)
                if round(self.predict(X[i])[0]) == round(y[i][0])
            )
            accuracy = correct / n

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                bar = "█" * int(accuracy * 20) + "░" * (20 - int(accuracy * 20))
                print(f"Epoch {epoch+1:>5}/{epochs}  loss={epoch_loss:.4f}  acc=[{bar}] {accuracy*100:.1f}%")

            # snapshot weights for visualization
            if epoch % snapshot_every == 0 or epoch == epochs - 1:
                # average activation per neuron across all samples
                avg_acts = []
                for li in range(len(self.layers)):
                    layer_avg = [
                        sum(all_acts[si][li][ni] for si in range(n)) / n
                        for ni in range(self.layers[li])
                    ]
                    avg_acts.append(layer_avg)

                # flatten weight magnitudes per layer for heatmap
                weight_magnitudes = []
                for W in self.weights:
                    flat = [abs(W[j][k]) for j in range(len(W)) for k in range(len(W[0]))]
                    weight_magnitudes.append(flat)

                self.history.append({
                    "epoch":      epoch + 1,
                    "loss":       round(epoch_loss, 6),
                    "accuracy":   round(accuracy, 4),
                    "activations": avg_acts,
                    "weight_mags": weight_magnitudes,
                    # sample a few predictions for display
                    "predictions": [
                        {
                            "input":  X[i],
                            "target": y[i],
                            "output": [round(v, 3) for v in self.predict(X[i])],
                        }
                        for i in range(min(4, n))
                    ],
                })

        self._is_trained = True
        self._X = X
        self._y = y
        return self

    # ── visualization ─────────────────────────────────────────────────────────

    def visualize(self):
        if not self._is_trained:
            raise RuntimeError("Call .train() first!")

        import os, sys
    
        # find the root vistify folder no matter where files are
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)  # goes up from ai/ to vistify/
        sys.path.insert(0, root_dir)
    
        from ai.training_template import render_training_html
    
        frames = self.history
        metadata = {
        "type":       "training",
        "layers":     self.layers,
        "activation": self.activation_name,
        "epochs":     frames[-1]["epoch"] if frames else 0,
    }
    
        html = render_training_html(frames, metadata)
    
        try:
            from IPython.display import display, HTML
            display(HTML(html))
        except ImportError:
            print(html)
    
        return html
