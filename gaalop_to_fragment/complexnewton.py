import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def complex_newton_step(z, poly):
    """Performs a single Newton iteration for a given polynomial."""
    p = Polynomial(poly)         # Create polynomial object
    dp = p.deriv()               # Compute derivative
    return z - p(z) / dp(z)      # Newton's step: z_n+1 = z_n - f(z)/f'(z)

# Define polynomial coefficients (z³ - 1 → [1, 0, 0, -1])
poly_coeffs = [1, 0, 0, -1]  # Represents f(z) = z³ - 1

# Generate a grid in the complex plane
x_vals = np.linspace(-2, 2, 200)  # Real part range
y_vals = np.linspace(-2, 2, 200)  # Imaginary part range
X, Y = np.meshgrid(x_vals, y_vals)
Z = X + 1j * Y  # Convert to complex numbers

# Compute Newton steps
Z_next = complex_newton_step(Z, poly_coeffs)
dZ=Z_next-Z
dZ/=np.abs(dZ)
U = np.real(dZ)  # Change in real part
V = np.imag(dZ)  # Change in imaginary part

# Plot vector field
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=100, color="blue")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("Vector Field of Newton's Method for Polynomial f(z)")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.show()