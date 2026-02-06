# src/core/parameters.py

from core.populationNode import PopulationNode


class Parameter(PopulationNode):
    """
    Learnable parameter (e.g., synaptic weight vector).

    Differences from PopulationNode:
      - Parameters are meant to persist across steps and be updated by optimizers.
      - .step(lr) applies a gradient descent update using accumulated .grad.
    """

    def __init__(self, data):
        # Parameters always require gradients
        super().__init__(data, requires_grad=True)

    def step(self, lr: float) -> None:
        """
        In-place parameter update: theta <- theta - lr * grad

        Assumes grad has already been computed via backprop.
        """
        if not self.requires_grad:
            return
        if len(self.grad) != len(self.data):
            raise ValueError("Parameter.grad and Parameter.data must have the same length.")

        lr = float(lr)
        for i in range(len(self.data)):
            self.data[i] -= lr * self.grad[i]
