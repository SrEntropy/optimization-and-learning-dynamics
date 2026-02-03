from learning_dynamics.core.parameter import Parameter
from learning_dynamics.core.optim import Momentum


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_momentum_beta_zero_equals_gd_update():
    """
    If beta = 0, then:
        v <- -lr * grad
        w <- w + v = w - lr * grad
    So it should behave like plain GD for that step.
    """
    w = Parameter([1.0, 2.0])
    w.grad = [1.0, 1.0]

    opt = Momentum([w], lr=0.1, beta=0.0)
    opt.step()

    assert close(w.data, [0.9, 1.9])


def test_momentum_two_steps_matches_hand_computation():
    """
    Hand-check 2 steps with constant gradient to verify velocity accumulation.

    Let w0 = 1.0, grad = 1.0, lr = 0.1, beta = 0.9
    v0 = 0

    Step 1:
      v1 = 0.9*v0 - 0.1*1 = -0.1
      w1 = w0 + v1 = 0.9

    Step 2:
      v2 = 0.9*v1 - 0.1*1 = 0.9*(-0.1) - 0.1 = -0.19
      w2 = w1 + v2 = 0.9 - 0.19 = 0.71
    """
    w = Parameter([1.0])
    opt = Momentum([w], lr=0.1, beta=0.9)

    # Step 1
    w.grad = [1.0]
    opt.step()
    assert close(w.data, [0.9])

    # Step 2 (same grad)
    w.grad = [1.0]
    opt.step()
    assert close(w.data, [0.71])


def test_momentum_zero_grad_clears_param_grads_only():
    """
    zero_grad should clear Parameter.grad.
    It should NOT reset velocity unless you explicitly add such behavior.
    """
    w = Parameter([1.0])
    opt = Momentum([w], lr=0.1, beta=0.9)

    w.grad = [1.0]
    opt.step()  # now velocity is non-zero

    w.grad = [5.0]
    opt.zero_grad()
    assert w.grad == [0.0]


if __name__ == "__main__":
    # Run without pytest
    test_momentum_beta_zero_equals_gd_update()
    test_momentum_two_steps_matches_hand_computation()
    test_momentum_zero_grad_clears_param_grads_only()
    print("All Momentum tests passed.")
