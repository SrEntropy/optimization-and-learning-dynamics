# Core Architecture Overview
This directory implements a minimal, neuro‑inspired autodiff engine built around Population Node, nodes representing homogeneous neural populations.

The system separates:
- state (PopulationNode)
- parameters (Parameter)
- operations (ops.py)
- learning rules (optim.py)


## PopulationNode
**File**: population.py
- Represents a passive state variable in the computation graph.
Think: neuronal activity, not synaptic weights.

## Parameter
**File**: parameter.py
- A specialized PopulationNode that represents learnable weights.

Think: synaptic weights.

### ops.py
- Defines all differentiable operations on PopulationNodes.
Think: mathematical transformations, not learning.

### optim.py
- Implements optimizers (currently Gradient Descent).

Think: learning schedule, not math.

### Training Loop (conceptual)
Build computation graph


### Design:
- PopulationNodes carry signals
- Parameters carry weights
- ops.py  defines math
- optim.py  defines learning
- backprop stitches everything togethe