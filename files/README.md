# visify

**Modern interactive visualizations for AI, algorithms, and math.**

```python
from visify import visualize

# Neural network — walk through forward pass layer by layer
visualize(model="neural_network", layers=[4, 8, 8, 2])

# With real activations from your input
visualize(model="neural_network", layers=[4, 8, 8, 2], input=x)

# Auto-introspect a PyTorch model
visualize(model=my_pytorch_model, input=x_tensor)

# Algorithm step-by-step
visualize(sort="quick", data=[5, 2, 9, 1, 7, 3])
```

Works in **VS Code notebooks**, **Jupyter Lab/Notebook**, and any IPython environment.

---

## Install

```bash
git clone <repo>
cd visify
pip install -e .
```

No required dependencies. IPython/Jupyter is detected automatically.

---

## Architecture

```
visify/
├── core/
│   ├── api.py          # visualize() — single entry point
│   ├── registry.py     # plugin map: keyword → visualizer class
│   └── base.py         # BaseVisualizer, Frame
├── ai/
│   └── nn.py           # NeuralNetVisualizer
├── algorithms/
│   └── sort.py         # SortVisualizer (quick, bubble, merge)
├── math/               # (coming soon)
└── render/
    ├── output.py       # RenderOutput — _repr_html_() for notebooks
    └── templates/
        └── nn_template.py   # HTML/JS visualization
```

Each visualizer is a generator that yields `Frame` dicts.  
The render engine consumes frames and produces HTML.

---

## Plugin API

Register your own visualizer in 10 lines:

```python
from visify import Registry
from visify.core.base import BaseVisualizer, Frame

class GradientDescentVisualizer(BaseVisualizer):
    def frames(self):
        data = self.kwargs.get("data", [])
        yield Frame(type="gd", label="Starting position", data=data)
        # ... yield more frames

Registry.register("gd", GradientDescentVisualizer)

# now this works:
visualize(gd="sgd", data=loss_surface, lr=0.01)
```

---

## Roadmap

- [x] Neural network forward pass
- [x] Quicksort / bubble sort step-by-step
- [ ] Live training visualizer (loss + accuracy streaming)
- [ ] Attention map (transformer)  
- [ ] Gradient descent on loss surface
- [ ] Graph algorithms (Dijkstra, BFS, DFS)
- [ ] Math: function plots, vector fields, transforms
- [ ] Export to SVG / GIF / MP4
