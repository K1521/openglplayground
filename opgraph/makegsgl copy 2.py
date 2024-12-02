import opgraph3 as opg

import sys
sys.path.append('./')
import algebra.dcga as dcga



t=dcga.Translator(3,4,5)
print((t*t.reverse()).blades)
def sanwich(V,m):
    return V*m*dcga.inverse(V)

xyz=(opg.VarNode("x"),opg.VarNode("y"),opg.VarNode("z"))
point=dcga.point(*xyz)
obj=dcga.toroid(2,0.5)
#obj=dcga.Plane(1,1,1,1)

obj=sanwich(t,obj)
iprod=point.inner(obj)
e=sum(abs(x) for x in iprod.blades.values())
ep=opg.EndpointNode([e])
#e=opg.EndpointNode([x*x+y*y+z*z-1])

#print(e.maxdepth())
#e.simplify()
ep.replacenode(lambda x:x)#subexpressionelimination
ep.replacenode(lambda x:x.normalizenode())#normlization
ep.replacenode(lambda x:x.consteliminationnode())#simplify
ep.mergenodes()
ep.replacenode(lambda x:x.consteliminationnode())


e=opg.EndpointNode(ep.backpropergation(xyz)+[ep.parents[0]])
e.replacenode(lambda x:x)#subexpressionelimination
e.replacenode(lambda x:x.normalizenode())#normlization
e.replacenode(lambda x:x.consteliminationnode())#simplify
e.mergenodes()
ep.replacenode(lambda x:x.consteliminationnode())

plan=e.asplanstr(compact=True).replace("node","n").split("\n")

last=list((plan.pop().split("Endpoint")[1].strip("()").replace(" ","").split(",")))


fundefspy="""
import numpy as np
sign=np.sign

"""
planpy="\n".join([f"    {l}" for l in plan])

#last="\nreturn ("+"+".join([f"{l}" for l in last])+");"
#last="\nreturn sqrt("+"+".join([f"abs({l})" for l in last])+");"
lastpy="\n    return ("+",".join(f"{l}" for l in last)+")"

fundec=fundefspy+f"def scene(p):\n    x,y,z=p\n{planpy+lastpy}\n"
with open("./scene2.py","w")as f:
    f.write(fundec)
print(fundec)
print()