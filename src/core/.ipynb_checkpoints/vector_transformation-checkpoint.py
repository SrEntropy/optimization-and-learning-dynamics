import numpy as np 
import pyvista as pv
# Example: 3D Scatter Plot

original_v = np.array([1,0,0])
# Example of tensor transformation:
## a 3*3 array = rank-2 tensor
## matrix transformation 
### Multiplying by the tensor rotates any vector in 3D space.
#### This tensor encodes how the coordinate axes are oriented in space.

M1 = np.array([
    [0.707, -0.707, 0],
    [0.707,  0.707, 0],
    [0,      0,     1]
])

# After transformation, the orignal V is reoriented based on the M1 tensor transformation
vectorT = M1 @ original_v

def plot_arrows(plotter, vector, colors):
    origin = np.array([[0,0,0]])
    direction = np.array([vector])   # must be 2D
    plotter.add_arrows(origin, direction, color=colors)

plotter = pv.Plotter()
plot_arrows(plotter, original_v, 'green')
plotter.show()

