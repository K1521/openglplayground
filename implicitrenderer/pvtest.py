
import pyvista as pv
import numpy as np


points = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]])

mesh = pv.PolyData(points)

faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]])
mesh.faces=faces
#mesh = pv.PolyData(points, faces)

# Visualize the
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="red",lighting=True,show_vertices=False)

plotter.show()