import opgraph3 as opg

import sys
sys.path.append('./')
import algebra.dcga as dcga

x,y,z=(opg.VarNode("x"),opg.VarNode("y"),opg.VarNode("z"))
#point=dcga.point(x,y,z)
# torus=dcga.toroid(2,0.5)
# iprod=point.inner(torus)
# e=opg.EndpointNode(iprod.blades.values())
e=opg.EndpointNode([x*x+y*y+z*z-1])

print(e.maxdepth())
#e.simplify()
e.replacenode(lambda x:x)#subexpressionelimination
e.replacenode(lambda x:x.normalizenode())#normlization
e.replacenode(lambda x:x.constelimination())#simplify
e.mergenodes()
e.replacenode(lambda x:x.constelimination())

plan=e.asplanstr(compact=True).replace("node","n").split("\n")

last=list(set(plan.pop().split("Endpoint")[1].strip("()").replace(" ","").split(",")))
plan="\n".join([f"float {l};" for l in plan])
last="\nreturn sqrt("+"+".join([f"abs({l})" for l in last])+");"
#last="\nreturn ("+"+".join([f"({l})" for l in last])+");"

fundec=f"float scene(vec3 p){{\nfloat x=p.x;float y=p.y;float z=p.z;\n{plan+last}\n}}"
with open("./scene.glsl","w")as f:
    f.write(fundec)
print(fundec)
print()