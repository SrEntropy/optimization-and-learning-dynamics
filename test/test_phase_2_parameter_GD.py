"""Phase 2 tests: Parameter + GD optimizer semantics.

Run with:
    python -m pytest -q
or just:
    python test_phase_2_parameter_GD.py
"""

from core.parameter import Parameter
from core.populationNode import PopulationNode
from core.optim import GD


def test_parameter_is_populationnode():
    w = Parameter([1.0, 2.0, 3.0])
    assert isinstance(w, PopulationNode)
    assert w.data == [1.0, 2.0, 3.0]
    assert w.grad == [0.0, 0.0, 0.0]


def test_zero_grad():
    w = Parameter([1.0, 2.0, 3.0])
    w.grad = [0.5, -0.3, 1.2]
    w.zero_grad()
    assert w.grad == [0.0, 0.0, 0.0]


def test_parameter_step_gd_update():
    w = Parameter([1.0, -2.0])
    w.grad = [0.1, -0.2]
    w.step(lr=0.5)
    assert w.data == [0.95, -1.9]


def test_optimizer_updates_all_params():
    w1 = Parameter([1.0])
    w2 = Parameter([2.0])
    w1.grad = [1.0]
    w2.grad = [2.0]

    opt = GD([w1, w2], lr=0.1)
    opt.step()

    assert w1.data == [0.9]
    assert w2.data == [1.8]


def test_optimizer_zero_grad():
    w1 = Parameter([1.0])
    w2 = Parameter([2.0])
    w1.grad = [1.0]
    w2.grad = [2.0]

    opt = GD([w1, w2], lr=0.1)
    opt.zero_grad()

    assert w1.grad == [0.0]
    assert w2.grad == [0.0]


def run_stability(lr, a=1.0, steps=10, theta0=5.0):
    """Simple stability tracer for L(θ)=0.5*a*θ^2: θ_{t+1}=(1-lr*a)θ_t"""
    w = Parameter([theta0])
    opt = GD([w], lr=lr)

    trace = [w.data[0]]
    for _ in range(steps):
        w.zero_grad()
        w.grad[0] = a * w.data[0]   # dL/dθ = aθ
        opt.step()
        trace.append(w.data[0])
    return trace


def test_stability_regions():
    # a = 1.0 -> stable iff 0 < lr < 2
    # lr=0.3 should shrink monotonically (r=0.7)
    t1 = run_stability(lr=0.3, a=1.0, steps=8)
    assert abs(t1[-1]) < abs(t1[0])

    # lr=1.5 should oscillate but shrink (r=-0.5)
    t2 = run_stability(lr=1.5, a=1.0, steps=8)
    assert abs(t2[-1]) < abs(t2[0])
    assert any(t2[i] * t2[i+1] < 0 for i in range(len(t2)-1))  # sign flip

    # lr=-0.2 is ascent (r=1.2) -> grows
    t3 = run_stability(lr=-0.2, a=1.0, steps=8)
    assert abs(t3[-1]) > abs(t3[0])

    # lr=2.2 unstable (r=-1.2) -> oscillate and grow
    t4 = run_stability(lr=2.2, a=1.0, steps=8)
    assert abs(t4[-1]) > abs(t4[0])
    assert any(t4[i] * t4[i+1] < 0 for i in range(len(t4)-1))  # sign flip


if __name__ == "__main__":
    # Allow running without pytest
    test_parameter_is_populationnode()
    test_zero_grad()
    test_parameter_step_gd_update()
    test_optimizer_updates_all_params()
    test_optimizer_zero_grad()
    test_stability_regions()
    print("All Phase 2 tests passed.")
