

import numpy as np
from core.parameter import Parameter
from core.populationNode import PopulationNode
from core.ops import mul, sum_pop, add
from models.activations import tanh, sigmoid, relu, softmax


class Neuron:
    """
    One neuron: y = tanh(w·x + b)

    - x is a PopulationNode vector
    - w is a Parameter vector
    - b is a Parameter scalar (len==1)
    - output is a PopulationNode scalar (len==1)
    """

    def __init__(self, n_inputs: int, activation: str = "tanh", seed: int | None = None):
        if not isinstance(n_inputs, int) or n_inputs <= 0:
            raise ValueError("n_inputs must be a positive int")

        self.activation = activation

        rng = np.random.default_rng(seed)
        # Use Parameter so grads flow + optimizer can update
        self.w = Parameter(rng.normal(size=(n_inputs,)).tolist())
        self.b = Parameter([float(rng.normal())])  # scalar Parameter

  
    def __call__(self, x: PopulationNode) -> PopulationNode:
        if not isinstance(x, PopulationNode):
            x = PopulationNode(x, requires_grad=False)

        if len(x.data) != len(self.w.data):
            raise ValueError(f"Input length {len(x.data)} != expected {len(self.w.data)}")

        # dot = sum_i (w_i * x_i)  -> scalar node
        dot = sum_pop(mul(self.w, x))        # scalar
        z = add(dot, self.b)                # scalar

        if self.activation == "linear":
            return z
        return tanh(z)

    def parameters(self):
        return [self.w, self.b]

        
    