# TODO: GD, Momentum

class GD:
    """
    Gradient Descent optimizer.
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


class Momentum:
    def __init__(self, params, lr, beta):
        self.params = params
        self.lr = lr
        self.beta = beta
        # Velocity same shape as p.data (list)
        self.v = [[0.0 for _ in p.data] for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            # v = beta*v - lr*grad
            for j in range(len(p.data)):
                self.v[i][j] = self.beta * self.v[i][j] - self.lr * p.grad[j]
                # p = p + v   (elementwise)
                p.data[j] += self.v[i][j]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
