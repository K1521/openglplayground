import sympy as sy

x,a,b,c,d,e=sy.symbols("x,a,b,c,d,e")

f=(x**4*a+x**3*b+x**2*c+x*d+e)
f_=(4*x**3*a+3*x**2*b+2*x*c+d)
print(sy.expand_complex(f/f_))