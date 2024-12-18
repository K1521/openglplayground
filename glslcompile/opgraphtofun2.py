

import opgraph as opg
#import math
from glslprog import glslprogrammpart
from types import SimpleNamespace

import sys
sys.path.append('./')
import algebra.dcga as dcga


def funtovisualize(x,y,z):


    t=dcga.Translator(3,4,5)
    #print((t*t.reverse()).blades)
    def sanwich(V,m):
        return V*m*dcga.inverse(V)

    #xyz=(opg.VarNode("x"),opg.VarNode("y"),opg.VarNode("z"))
    point=dcga.point(x,y,z)
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,1)

    #obj=sanwich(t,obj)
    iprod=point.inner(obj)
    return list(iprod.blades.values())
    #e=sum(abs(x) for x in iprod.blades.values())


def funtovisualize(x,y,z):
    return [(x*x+y*y-1)*(x*x+z*z-1)]

def strtopyfun(funstring):
    local_vars = {}
    exec(funstring, globals(), local_vars)
    return SimpleNamespace(**local_vars)

def makefuntailorglsl(endpoint):
    plan=endpoint.asplanstr(compact=True).replace("node","n").split("\n")
    last=list((plan.pop().split("Endpoint")[1][1:-1].replace(" ","").split(",")))

    plangsgl="\n".join([f"double {l};" for l in plan])

    #last="\nreturn ("+"+".join([f"{l}" for l in last])+");"
    #last="\nreturn sqrt("+"+".join([f"abs({l})" for l in last])+");"
    #lastgsgl="return vec4("+",".join(l for l in last)+");"
    polyset=[]
    numparams=5
    numpolys=len(last)//numparams
    for param in range(numparams):
        for poly in range(numpolys):
            polyset.append(f"polys[{poly}][{param}]={last[param*numpolys+poly]};")
    #[[p1,p2][][]]
    polyset='\n'.join(polyset)
    header=f"""
const int numparams={numparams};
const int numpolys={numpolys};
const int degree=numparams-1;
double[numpolys][numparams] polys;
void compilepolys(vec3 p,vec3 d);

"""
    body=f"""
void compilepolys(vec3 p,vec3 d){{
double ox=p.x;
double oy=p.y;
double oz=p.z;
double dx=d.x;
double dy=d.y;
double dz=d.z;
{plangsgl}
{polyset}
}}


"""

    return glslprogrammpart(header,body,bodyprio=-10,headerprio=10)

def makefuntailorpy(endpoint):
    plan=endpoint.asplanstr(compact=True).replace("node","n").split("\n")
    last=list((plan.pop().split("Endpoint")[1][1:-1].replace(" ","").split(",")))

    planpy="\n".join([f"    {l}" for l in plan])

    #last="\nreturn ("+"+".join([f"{l}" for l in last])+");"
    #last="\nreturn sqrt("+"+".join([f"abs({l})" for l in last])+");"
    #lastgsgl="return vec4("+",".join(l for l in last)+");"
    #[[p1,p2][][]]
    fundec=f"""
def calcpoly(x):
    return poly[0]+poly[1]*x+poly[2]*x**2+poly[3]*x**3+poly[4]*x**3
    
def compilepolys(p,d):
    a=0
    ox=p[0]
    oy=p[1]
    oz=p[2]
    dx=d[0]
    dy=d[1]
    dz=d[2]
{planpy}
    return {",".join(last)}



"""

    return fundec


def makefuntailor(f):
    ox,oy,oz,dx,dy,dz,a=[opg.VarNode(name) for name in "ox,oy,oz,dx,dy,dz,a".split(",")]
    x,y,z=ox+a*dx,oy+a*dy,oz+a*dz#defines a ray in 3d o+a*d where o is the start and d is the direction


    funcgraphoutputs=f(x,y,z)# 
    funcgraphoutputs=opg.simplify(funcgraphoutputs)
    #print(endpoint.asplanstr(compact=True).replace("node","n"))


    derivatives=[funcgraphoutputs]
    for i in range(1,5):
        #print( type(derivatives[-1][0]))
        
        derivatives.append([p.backpropergation(a)/i for p in derivatives[-1]])
        #TODO use forward mode auto diff. should be faster and simpler


    derivflatt=opg.EndpointNode([param for deriv in derivatives for param in deriv])
    derivflatt.replacenode(lambda node:opg.ConstNode(0) if node.signature()==opg.VarNode("a").signature() else node)
    derivflatt=opg.simplify(derivflatt)

    fundecglsl=makefuntailorglsl(derivflatt)
    pyfun=makefuntailorpy(derivflatt)
    #print(fundec)
    #print(longendpoint.asplanstr(compact=True).replace("node","n"))
    #print(makefuntailorpy(longendpoint))
    return fundecglsl,pyfun