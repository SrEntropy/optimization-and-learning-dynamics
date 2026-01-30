from learning_dynamics.core.populationNode import PopulationNode


class Parameter(PopulationNode):
    """
    Learnable parameter (synaptic weight vector).

    - .data : persistent weight values
    - .grad : accumulated gradients from backprop
    - updates happen only when an optimizer calls .step()
    - unlike PopulationNode, this object *does* learn
    """

    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = [0.0 for _ in self.grad]

    def step(self, lr):
        if not self.requires_grad:
            return
        for i in range(len(self.data)):
            self.data[i] -= lr * self.grad[i]
