from core.populationNode import PopulationNode
from models.layer import Layer
from models.neuron import Neuron

class MLP:
    def __init__(self, n_inputs: int, layer_sizes, activation: str = "tanh", seed: int | None = None):
        sizes = [n_inputs] + list(layer_sizes)
        self.layers = []
        for i in range(len(layer_sizes)):
            act = activation if i < len(layer_sizes) - 1 else "linear"  # often linear last layer
            self.layers.append(Layer(sizes[i], sizes[i+1], activation=act, seed=None if seed is None else seed + 100*i))

    def __call__(self, x: PopulationNode) -> PopulationNode:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params