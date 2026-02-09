from core.ops import stack
from core.populationNode import PopulationNode
from models.neuron import Neuron

class Layer:
    """A layer is a list of neurons producing a vector output."""

    def __init__(self, n_inputs: int, n_outputs: int, activation: str = "tanh", seed: int | None = None):
        self.neurons = [
            Neuron(n_inputs, activation=activation, seed=None if seed is None else seed + i)
            for i in range(n_outputs)
        ]

    def __call__(self, x: PopulationNode) -> PopulationNode:
        outs = [n(x) for n in self.neurons]  # list of scalar nodes
        return stack(outs)                   # vector node

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

