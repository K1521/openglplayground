import sympy as sp

# Define the symbols
x, y, z, r = sp.symbols('x y z r')

# List of the polynomials
polynomials = [
    x**2, x*y, x*z, r*x/2 - x/2, r*x/2 + x/2, x*y, y**2, y*z, 
    r*y/2 - y/2, r*y/2 + y/2, x*z, y*z, z**2, r*z/2 - z/2, 
    r*z/2 + z/2, r*x/2 - x/2, r*y/2 - y/2, r*z/2 - z/2, 
    r**2/4 - r/2 + 1/4, r**2/4 - 1/4, r*x/2 + x/2, r*y/2 + y/2, 
    r*z/2 + z/2, r**2/4 - 1/4, r**2/4 + r/2 + 1/4
]

print(bin(15))

# Create a matrix where each row represents the coefficients of the terms in each polynomial
matrix = sp.Matrix([[sp.simplify(poly) for poly in polynomials]])
print(matrix)
# Find a basis using row reduction
basis = matrix.row_join(matrix.T).rref()[0]  # Using RREF to find linearly independent rows

# Display the basis
print("Basis for the given set of polynomials:")
for poly in basis:
    print(poly)
