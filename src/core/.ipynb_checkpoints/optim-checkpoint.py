# learning_dynamics/core/optim.py

from typing import List


class GD:
    """
    Gradient Descent optimizer.

    Expects parameters with:
      - .data (list[float])
      - .grad (list[float])
      - .step(lr)
      - .zero_grad()
    """

    def __init__(self, params: List, lr: float):
        self.params = list(params)
        self.lr = float(lr)

    def step(self) -> None:
        for p in self.params:
            # Let Parameter.step handle requires_grad, but it's fine to guard here too.
            if getattr(p, "requires_grad", True):
                p.step(self.lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class Momentum:
    """
    Momentum optimizer (heavy-ball style):

      v <- beta*v - lr*grad
      theta <- theta + v

    Stored velocity has same shape as each parameter's .data.
    """

    def __init__(self, params: List, lr: float, beta: float = 0.9):
        self.params = list(params)
        self.lr = float(lr)
        self.beta = float(beta)

        # Velocity buffers: one list per parameter
        self.v = [[0.0 for _ in p.data] for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if not getattr(p, "requires_grad", True):
                continue

            if len(p.grad) != len(p.data):
                raise ValueError("Param.grad and Param.data must match in length.")

            # v = beta*v - lr*grad
            # p = p + v
            for j in range(len(p.data)):
                self.v[i][j] = self.beta * self.v[i][j] - self.lr * p.grad[j]
                p.data[j] += self.v[i][j]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
