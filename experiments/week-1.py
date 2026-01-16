# week1_population_experiments.py
# Core gradient-flow micro-experiments (no training)
# See week1_experiments_explained.md for interpretation.

from learning_dynamics.core.populationNode import PopulationNode
from learning_dynamics.core.ops import tanh, sum_pop

def experiment_1():
    print("\n[Experiment 1] Gradient Shape Through Nonlinearity")
    xs = [-6, -3.0, -0.5, 0, 0.5, 3.0, 6]
    x = PopulationNode(xs)
    y = tanh(x)
    sum_pop(y).backprop()
    print("Input values:", xs)
    print("Gradients:   ", x.grad)


def experiment_2():
    print("\n[Experiment 2] Depth-Induced Gradient Decay")
    inputs = [0.5, 4.5]
    for val in inputs:
        x = PopulationNode(val)
        y = x
        for _ in range(8):
            y = tanh(y)
        y.backprop()
        print("Input:", x.data)
        print("Gradient at input:", x.grad)


def experiment_3():
    print("\n[Experiment 3] Population vs Scalar Sensitivity")
    
    x_scalar = PopulationNode(0.5)
    y_scalar = tanh(x_scalar)
    y_scalar.backprop()
    print(f"Scalar = {x_scalar.data} gradient sum:", sum(x_scalar.grad))

    x_pop = PopulationNode([0.5, 0.5, 0.5, 0.5])
    y_pop = tanh(x_pop)
    sum_pop(y_pop).backprop()
    print(f"Population = {x_pop.data} gradient sum:", sum(x_pop.grad))


def experiment_4():
    print("\n[Experiment 4] Shared Subgraph / Credit Assignment")
    x = PopulationNode([1.0, 2.0, 3.0])
    y = x * x + x
    z = sum_pop(y)
    z.backprop()
    print("Input:", x.data)
    print("Gradient:", x.grad)


def experiment_5():
    print("\n[Experiment 5] Reduction as an Information Bottleneck")
    inputs = [[1.0, 1.0, 1.0],[1.0, 1.0, 3.0]]
    for i in inputs:
        x = PopulationNode(i)
        y = tanh(x)
        sum_pop(y).backprop()
        print(f"Population = {x.data}, Sum of gradients:", sum(x.grad))


def experiment_6():
    print("\n[Experiment 6] Gradient Symmetry Breaking")
    x = PopulationNode([0.5, 0.5001])
    y = tanh(x)
    sum_pop(y).backprop()
    print("Input:", x.data)
    print("Gradients:", x.grad)


def experiment_7():
    print("\n[Experiment 7] Gradient Flow Without Updates")
    for v in [-3, -1, 0, 1, 3]:
        x = PopulationNode(v)
        y = tanh(x)
        y.backprop()
        print(f"Input {v}: gradient {x.grad}")


def main():
    print("=== WEEK 1 — POPULATION GRADIENT MICRO-EXPERIMENTS ===")
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6()
    experiment_7()


if __name__ == "__main__":
    main()
