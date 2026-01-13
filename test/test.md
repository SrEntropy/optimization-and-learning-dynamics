# 🧪 PopulationTensor Autodiff Engine — Test Suite

A collection of tests validating correctness of gradients, population semantics, and backpropagation behavior in the `PopulationTensor` autodiff engine.

---

## **Test 1 — Scalar Chain Rule**

**Expression:** 
- x = 2.0
- y = 4.0

$z = x\cdot y + x$


**Expected gradients:**  

$ \frac{\partial z}{\partial x} = y + 1 = 4$   

$\frac{\partial z}{\partial y} = x = 2 $

**Backprop Trace:**
~~~
- [NODE] op=+, value=[8.0], grad=[1.0] | <-- Parents=[[6.0], [2.0]]
- [NODE] op=*, value=[6.0], grad=[1.0] | <-- Parents=[[2.0], [3.0]]
- [NODE] op=leaf, value=[3.0], grad=[2.0] | <-- Parents=[]
- [NODE] op=leaf, value=[2.0], grad=[4.0] | <-- Parents=[] 
~~~
**Result:**  
- **x.grad:** `[4.0]`  
- **y.grad:** `[2.0]`  

**✓ Passed — gradients match analytical derivatives**

---

## **Test 2 — Vector Chain Rule**

**Expression:**  
- x = [1.0, 2.0]
- y = [3.0, 4.0]

$z = \sum (x + y)$

**Expected gradients:**  
$\frac{\partial z}{\partial x_i} = 1 $

$\frac{\partial z}{\partial y_i} = 1 $

**Backprop Trace:**
```
- [NODE] op=sum, value=[10.0], grad=[1.0] | <-- Parents=[[4.0, 6.0]]
- [NODE] op=+, value=[4.0, 6.0], grad=[1.0, 1.0] | <-- Parents=[[1.0, 2.0], [3.0, 4.0]]
- [NODE] op=leaf, value=[3.0, 4.0], grad=[1.0, 1.0] | <-- Parents=[]
- [NODE] op=leaf, value=[1.0, 2.0], grad=[1.0, 1.0] | <-- Parents=[]
```

**Result:**  
- **x.grad:** `[1.0, 1.0]`  
- **y.grad:** `[1.0, 1.0]`  

**✓ Passed — vector gradients match analytical derivatives**

---

## **Test 3 — XOR‑Ready Neuron (tanh activation)**

**Expression:**  
- w = [1.0, -1.0]
- x = [1.0, 0.0]
- b = 0.0

$a = \tanh(w \cdot x + b)$

This test validates:
- population‑wise dot product  
- tanh derivative  
- correct gradient flow into weights, inputs, and bias  

**Backprop Trace:**
```
- [NODE] op=tanh, value=[0.7615941559557649], grad=[1.0] | <-- Parents=[[1.0]]
- [NODE] op=+, value=[1.0], grad=[0.41997434161402614] | <-- Parents=[[1.0], [0.0]]
- [NODE] op=leaf, value=[0.0], grad=[0.41997434161402614] | <-- Parents=[]
- [NODE] op=sum, value=[1.0], grad=[0.41997434161402614] | <-- Parents=[[1.0, -0.0]]
- [NODE] op=*, value=[1.0, -0.0], grad=[0.41997434161402614, 0.41997434161402614] | <-- Parents=[[1.0, -1.0], [1.0, 0.0]]
- [NODE] op=leaf, value=[1.0, 0.0], grad=[0.41997434161402614, -0.41997434161402614] | <-- Parents=[]
- [NODE] op=leaf, value=[1.0, -1.0], grad=[0.41997434161402614, 0.0] | <-- Parents=[]
```

**Result:**  
- **x.grad:** `[0.41997434161402614, -0.41997434161402614]`  
- **w.grad:** `[0.41997434161402614, 0.0]`  
- **b.grad:** `[0.41997434161402614]`  

**✓ Passed — tanh derivative and neuron gradients are correct**

---

