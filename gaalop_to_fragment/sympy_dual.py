

import sympy as sp

# Define epsilon as a symbol
epsilon = sp.symbols('epsilon')

# Define the third-order dual numbers
f = sp.symbols('f0 f1 f2 f3')  # f0 + f1*epsilon + f2*epsilon^2 + f3*epsilon^3
g = sp.symbols('g0 g1 g2 g3')  # g0 + g1*epsilon + g2*epsilon^2 + g3*epsilon^3

# Define f and g as third-order dual numbers
f_num = f[0] + f[1]*epsilon + f[2]*epsilon**2 + f[3]*epsilon**3
g_num = g[0] + g[1]*epsilon + g[2]*epsilon**2 + g[3]*epsilon**3

# Multiply f_num and g_num
product = sp.expand(f_num * g_num)

# Simplify (because epsilon^2 and higher powers vanish)
product_simplified = sp.simplify(product)
product_collected = sp.collect(product_simplified, epsilon)
# Show the result
print("Product of f and g:")
print(product_collected)