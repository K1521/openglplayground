import sys
sys.path.append('./')

import algebra.dcga as dcga
import sympy as sy
x,y,z,r=sy.symbols("x,y,z,r")
p=dcga.point(x,y,z)
plist=list(p.blades.values())
#r=x*x+y*x+z*z
for i in range(len(plist)):
    plist[i]=plist[i].simplify().subs(1,1.0).subs(x**2 + y**2 + z**2 + 1.0,r+1).subs(x**2 + y**2 + z**2 - 1.0,r-1)#.expand()
print(plist)
print(len(set(plist)),len(plist))

#x,y,z,(r-1),(r+1)