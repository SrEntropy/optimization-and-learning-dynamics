import math
import pytest

# ---------------------------------------------------------------------
# Import shim: adjust these if your package layout differs
# ---------------------------------------------------------------------
try:
    from core.populationNode import PopulationNode
    from core.parameters import Parameter
    from core.optim import GD, Momentum
    from core.ops import add, sub, mul, tanh, sum_pop, matvec
except Exception:
    # fallback if your package is namespaced differently
    from learning_dynamics.core.populationNode import PopulationNode
    from src.core.parameters import Parameter
    from learning_dynamics.core.optim import GD, Momentum
    from learning_dynamics.core.ops import add, sub, mul, tanh, sum_pop, matvec


# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------
EPS = 1e-5
ATOL = 1e-4
RTOL = 1e-4


def close_list(a, b, atol=ATOL, rtol=RTOL):
    assert len(a) == len(b)
    for i in range(len(a)):
        if not math.isclose(a[i], b[i], abs_tol=atol, rel_tol=rtol):
            raise AssertionError(f"Mismatch at idx={i}: {a[i]} vs {b[i]}")


def scalar_loss(fn, x_vals):
    """
    Evaluate a scalar loss by building a graph with a PopulationNode from x_vals
    and returning the final output node.

    fn must return a PopulationNode (scalar output or reducible to scalar).
    """
    x = PopulationNode(list(x_vals))
    out = fn(x)
    # force scalar objective if out is vector:
    if len(out.data) != 1:
        out = sum_pop(out)
    return x, out


def finite_diff_grad(fn, x0, eps=EPS):
    """
    Central finite differences for scalar objective L(x).
    x0 is list[float].
    """
    g = [0.0] * len(x0)
    for i in range(len(x0)):
        xp = list(x0)
        xm = list(x0)
        xp[i] += eps
        xm[i] -= eps

        _, out_p = scalar_loss(fn, xp)
        _, out_m = scalar_loss(fn, xm)

        Lp = out_p.data[0]
        Lm = out_m.data[0]
        g[i] = (Lp - Lm) / (2 * eps)
    return g


def engine_grad(fn, x0):
    x, out = scalar_loss(fn, x0)
    out.backprop()
    return list(x.grad), out


# ---------------------------------------------------------------------
# Forward correctness (basic sanity)
# ---------------------------------------------------------------------
def test_forward_add_mul_sub_tanh_sum():
    x = PopulationNode([1.0, 2.0, 3.0])
    y = PopulationNode([10.0, 20.0, 30.0])

    z1 = x + y
    assert z1.data == [11.0, 22.0, 33.0]

    z2 = x * y
    assert z2.data == [10.0, 40.0, 90.0]

    z3 = y - x
    assert z3.data == [9.0, 18.0, 27.0]

    z4 = x.tanh()
    assert all(-1.0 < v < 1.0 for v in z4.data)

    s = sum_pop(x)
    assert s.data == [6.0]


def test_forward_matvec():
    A = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    x = PopulationNode([5.0, 6.0])
    y = matvec(A, x)
    assert y.data == [1*5 + 2*6, 3*5 + 4*6]  # [17, 39]


# ---------------------------------------------------------------------
# Shared‑node correctness (multi‑path gradient flow)
# ---------------------------------------------------------------------

def test_shared_node_simple():
    """
    z = m + m where m = x * y
    Ensures correct accumulation on shared subgraphs.
    """
    x = PopulationNode([2.0], requires_grad=True)
    y = PopulationNode([3.0], requires_grad=True)

    m = x * y
    z = m + m   # shared node

    x.zero_grad()
    y.zero_grad()
    z.backprop()

    # dz/dx = 2*y = 6
    # dz/dy = 2*x = 4
    assert x.grad == [6.0]
    assert y.grad == [4.0]


def test_shared_node_deep():
    """
    z = m + (m * 2) where m = x * y
    Ensures correct multi‑path gradient flow.
    """
    x = PopulationNode([2.0], requires_grad=True)
    y = PopulationNode([5.0], requires_grad=True)

    m = x * y
    z = m + (m * 2.0)   # 3*m total

    x.zero_grad()
    y.zero_grad()
    z.backprop()

    # dz/dx = 3*y = 15
    # dz/dy = 3*x = 6
    assert x.grad == [15.0]
    assert y.grad == [6.0]


def test_shared_node_multiple_backward():
    """
    Ensures:
    - leaf nodes accumulate across backward calls
    - shared intermediate nodes DO NOT accumulate across calls
    """
    x = PopulationNode([2.0], requires_grad=True)
    y = PopulationNode([3.0], requires_grad=True)

    m = x * y
    z = m + m

    x.zero_grad()
    y.zero_grad()

    # First backward
    z.backprop()
    g1x, g1y = x.grad[:], y.grad[:]

    # Second backward (no zero_grad)
    z.backprop()
    g2x, g2y = x.grad[:], y.grad[:]

    # Leaves accumulate
    assert g2x == [2 * g1x[0]]
    assert g2y == [2 * g1y[0]]

    # Intermediate node m should NOT accumulate across calls
    # (implicitly tested because if it did, leaf grads would be wrong)
    assert g2x == [12.0]   # 6 → 12
    assert g2y == [8.0]    # 4 → 8


def test_shared_node_vector():
    """
    Shared node with vector outputs.
    v = [1,2,3]
    m = v * 2
    z = m + m
    dz/dv = 4
    """
    v = PopulationNode([1.0, 2.0, 3.0], requires_grad=True)

    m = v * 2.0
    z = m + m

    v.zero_grad()
    z.backprop()

    assert v.grad == [4.0, 4.0, 4.0]

# ---------------------------------------------------------------------
# Backward: gradient checks via finite differences
# ---------------------------------------------------------------------
def test_grad_check_add():
    fn = lambda x: sum_pop(x + PopulationNode([2.0, -1.0, 0.5], requires_grad=False))
    x0 = [0.2, -0.4, 1.7]
    g_eng, _ = engine_grad(fn, x0)
    g_fd = finite_diff_grad(fn, x0)
    close_list(g_eng, g_fd)


def test_grad_check_sub():
    fn = lambda x: sum_pop(PopulationNode([1.0, 2.0, 3.0], requires_grad=False) - x)
    x0 = [0.5, -1.0, 2.0]
    g_eng, _ = engine_grad(fn, x0)
    g_fd = finite_diff_grad(fn, x0)
    close_list(g_eng, g_fd)


def test_grad_check_mul():
    c = PopulationNode([3.0, -2.0, 0.25], requires_grad=False)
    fn = lambda x: sum_pop(x * c)
    x0 = [0.3, 0.4, -1.2]
    g_eng, _ = engine_grad(fn, x0)
    g_fd = finite_diff_grad(fn, x0)
    close_list(g_eng, g_fd)


def test_grad_check_tanh_chain():
    # L = sum(tanh(x) * x)
    fn = lambda x: sum_pop(x.tanh() * x)
    x0 = [0.1, -0.7, 1.3]
    g_eng, _ = engine_grad(fn, x0)
    g_fd = finite_diff_grad(fn, x0)
    close_list(g_eng, g_fd, atol=2e-4, rtol=2e-4)


def test_grad_check_matvec():
    A = [
        [0.3, 1.7, -2.0],
        [1.0, -0.5, 0.2],
    ]
    fn = lambda x: sum_pop(matvec(A, x))
    x0 = [0.4, -0.8, 1.1]
    g_eng, _ = engine_grad(fn, x0)
    g_fd = finite_diff_grad(fn, x0)
    close_list(g_eng, g_fd, atol=2e-4, rtol=2e-4)


# ---------------------------------------------------------------------
# Broadcasting semantics + backward reduction
# ---------------------------------------------------------------------
def test_broadcast_scalar_to_vector_add_and_backward():
    # x is scalar, v is vector
    x = PopulationNode(2.0)
    v = PopulationNode([1.0, 2.0, 3.0])
    out = sum_pop(x + v)  # scalar objective

    out.backprop()
    # d/dx sum_i (x + v_i) = 3
    assert x.grad == [3.0]
    # d/dv_i = 1
    assert v.grad == [1.0, 1.0, 1.0]


def test_broadcast_scalar_to_vector_mul_and_backward():
    x = PopulationNode(2.0)
    v = PopulationNode([1.0, 2.0, 3.0])
    out = sum_pop(x * v)

    out.backprop()
    # d/dx sum_i x*v_i = sum_i v_i = 6
    assert x.grad == [6.0]
    # d/dv_i = x = 2
    assert v.grad == [2.0, 2.0, 2.0]


def test_broadcast_incompatible_raises():
    a = PopulationNode([1.0, 2.0])
    b = PopulationNode([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _ = a + b  # cannot broadcast 2 <-> 3


# ---------------------------------------------------------------------
# requires_grad correctness for constants
# ---------------------------------------------------------------------
def test_constants_do_not_receive_grads():
    x = PopulationNode([1.0, 2.0, 3.0])
    c = PopulationNode([10.0, 10.0, 10.0], requires_grad=False)
    out = sum_pop(x * c)

    out.backprop()
    assert c.grad == [0.0, 0.0, 0.0]
    assert x.grad == [10.0, 10.0, 10.0]


# ---------------------------------------------------------------------
# Accumulation semantics + graph clearing
# ---------------------------------------------------------------------
def test_zero_grad_graph_clears_all_nodes():
    x = PopulationNode(2.0)
    y = PopulationNode(3.0)
    z = x * y + x  # scalar
    z.backprop()
    assert x.grad != [0.0] and y.grad != [0.0] and z.grad != [0.0]

    z.zero_grad_graph()
    assert x.grad == [0.0]
    assert y.grad == [0.0]
    assert z.grad == [0.0]


def test_backprop_accumulates_leaves_but_not_intermediates():
    """
    This test enforces your chosen semantic ("Approach A"):

    - Call backprop twice WITHOUT calling zero_grad_graph().
    - Leaf grads should accumulate (double).
    - Intermediate grads should NOT accumulate (they should reflect one-pass grads),
      because backprop resets intermediates each call.
    """
    x = PopulationNode(2.0)
    y = PopulationNode(3.0)
    a = x * y          # intermediate
    z = a + x          # output

    # Pass 1
    z.backprop()
    g1x, g1y = x.grad[:], y.grad[:]
    g1a = a.grad[:]

    # Pass 2 (no manual reset)
    z.backprop()
    g2x, g2y = x.grad[:], y.grad[:]
    g2a = a.grad[:]

    # Leaves accumulate:
    assert math.isclose(g2x[0], 2 * g1x[0], rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(g2y[0], 2 * g1y[0], rel_tol=1e-9, abs_tol=1e-9)

    # Intermediate does NOT accumulate (should be same after each pass)
    # For z = (x*y) + x, upstream into a is 1 each pass => a.grad should remain 1.
    assert math.isclose(g2a[0], g1a[0], rel_tol=1e-9, abs_tol=1e-9)


def test_backprop_seed_grad_vector_output_defaults_to_ones():
    # y = tanh(x) is vector; default seed means upstream ones => grad should match finite diffs of sum(y)
    fn = lambda x: x.tanh()  # vector output; scalar_loss will sum_pop it
    x0 = [0.2, -0.2, 0.7]
    g_eng, _ = engine_grad(fn, x0)
    g_fd = finite_diff_grad(fn, x0)
    close_list(g_eng, g_fd, atol=2e-4, rtol=2e-4)


# ---------------------------------------------------------------------
# Parameter + optimizer correctness
# ---------------------------------------------------------------------
def test_parameter_step_matches_gd_update():
    w = Parameter([1.0, -2.0, 0.5])
    x = PopulationNode([3.0, 4.0, 5.0], requires_grad=False)

    # L = sum(w * x)
    loss = sum_pop(w * x)
    loss.backprop()

    # grad_w should be x
    assert w.grad == [3.0, 4.0, 5.0]

    lr = 0.1
    w_old = w.data[:]
    w.step(lr)
    assert w.data == [w_old[i] - lr * w.grad[i] for i in range(3)]


def test_gd_optimizer_step_and_zero_grad():
    w = Parameter([1.0, 2.0])
    x = PopulationNode([10.0, -1.0], requires_grad=False)

    loss = sum_pop(w * x)
    loss.backprop()
    assert w.grad == [10.0, -1.0]

    opt = GD([w], lr=0.5)
    opt.step()
    assert w.data == [1.0 - 0.5 * 10.0, 2.0 - 0.5 * (-1.0)]

    opt.zero_grad()
    assert w.grad == [0.0, 0.0]


def test_momentum_matches_reference_update():
    w = Parameter([1.0, 2.0])
    # manually set grad for deterministic test
    w.grad = [3.0, -4.0]

    lr = 0.1
    beta = 0.9
    opt = Momentum([w], lr=lr, beta=beta)

    # step 1: v = -lr*grad, w += v
    opt.step()
    # v = [-0.3, 0.4], w = [0.7, 2.4]
    close_list(w.data, [0.7, 2.4], atol=1e-9, rtol=0.0)

    # step 2 with same grad: v = beta*v - lr*grad
    w.grad = [3.0, -4.0]
    opt.step()
    # v2 = 0.9*[-0.3, 0.4] - 0.1*[3,-4] = [-0.27-0.3, 0.36+0.4] = [-0.57, 0.76]
    # w2 = [0.7-0.57, 2.4+0.76] = [0.13, 3.16]
    close_list(w.data, [0.13, 3.16], atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------
# Shape enforcement checks
# ---------------------------------------------------------------------
def test_elementwise_shape_mismatch_raises():
    a = PopulationNode([1.0, 2.0, 3.0])
    b = PopulationNode([1.0, 2.0])
    with pytest.raises(ValueError):
        _ = a * b


def test_matvec_shape_mismatch_raises():
    A = [[1.0, 2.0], [3.0, 4.0]]
    x = PopulationNode([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        _ = matvec(A, x)
