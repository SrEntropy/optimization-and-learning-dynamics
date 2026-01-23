
from core.populationNode import PopulationNode


class Parameter(PopulationNode):
    """
    Learnable parameter (synaptic weight vector).

    - .data : persistent weight values
    - .grad : accumulated gradients from backprop
    - updates happen only when an optimizer calls .step()
    - unlike PopulationNode, this object *does* learn
    """

    def __init__(self, data):
        super().__init__(data)

    def zero_grad(self):
        self.grad = [0.0 for _ in self.grad]

    def step(self, lr):
        for i in range(len(self.data)):
            self.data[i]-= lr * self.grad[i]


