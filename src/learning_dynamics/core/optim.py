# TODO: GD, Momentum
class GD:
    """
    Gradient Descent optimizer.

    - holds a list of Parameter objects
    - zero_grad() clears their gradients
    - step() applies weight updates using stored gradients

    This object decides *when* learning happens,
    but parameters decide *how* they update.
    """

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.step(self.lr)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
