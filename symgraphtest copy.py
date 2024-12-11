import casadi as ca

# Define symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
z = ca.SX.sym('z')
a = ca.SX.sym('a')

# Define a function f(x, y, z, a)
y1 = x + y + z + a
y2 = x * y - z * a
outputs = ca.vertcat(y1, y2)

# Create a CasADi function
f = ca.Function('f', [x, y, z, a], [outputs])


f.disp(True)