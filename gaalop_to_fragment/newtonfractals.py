import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# Define polynomial f(z) = z^3 - 1
poly_coeffs = [1, 0, 0, -1]  # Represents z³ - 1
p = Polynomial(poly_coeffs)
dp = p.deriv()  # Compute derivative

# Compute actual roots
roots = np.roots(poly_coeffs)

# Grid setup
resolution = 500
x_vals = np.linspace(-2, 2, resolution)
y_vals = np.linspace(-2, 2, resolution)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X + 1j * Y  # Complex grid

# Newton Iteration Parameters
max_iters = 100
tolerance = 1e-6

# Store final root index and distances
final_root = np.zeros(Z.shape, dtype=int)
distances = np.zeros((Z.shape[0], Z.shape[1], len(roots)))

# Newton iteration loop
for i in range(1,max_iters):
    Z_prev = Z.copy()
    dZ = p(Z) / dp(Z)  # Newton iteration
    #dZ *= np.sqrt(np.abs(dZ))  # Normalize
    Z -= dZ#/max(1.2**(20-i),1)  # Newton iteration
    #if np.allclose(Z, Z_prev, atol=tolerance):  # Stop if converged
    #    break

# Assign each point to its closest root
for j, root in enumerate(roots):
    distances[..., j] = np.abs(Z - root)

final_root = np.argmin(distances, axis=-1)  # Get closest root index
barycentric = 1 - distances / np.sum(distances, axis=-1, keepdims=True)  # Normalize distances

# Define colors for each root
colors = np.array([
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
])

# Compute final interpolated color
fractal_colors = np.tensordot(barycentric, colors, axes=([2], [0]))

# Plot the fractal
plt.figure(figsize=(8, 8))
plt.imshow(fractal_colors, extent=(-2, 2, -2, 2))
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("Newton Fractal with Barycentric Color Interpolation")
plt.show()