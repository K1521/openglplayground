

import sys
sys.path.append('./')
import algebra.dcga as dcga
sys.path.append('./opgraph')
import opgraph3 as opg
import math





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


# def funtovisualize(x,y,z):


#     return [(x*x+y*y-1)*(x*x+z*z-1)]

def simplify(endpoint):
    endpoint.replacenode(lambda x:x)#subexpressionelimination
    endpoint.replacenode(lambda x:x.normalizenode())#normlization
    endpoint.replacenode(lambda x:x.consteliminationnode())#simplify
    endpoint.mergenodes()
    endpoint.replacenode(lambda x:x.consteliminationnode())

def stringifynodesglsl(endpoint):
    plan=endpoint.asplanstr(compact=True).replace("node","n").split("\n")
    last=list((plan.pop().split("Endpoint")[1][1:-1].replace(" ","").split(",")))

    plangsgl="\n".join([f"float {l};" for l in plan])

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
    fundec=f"""
const int numpolys={numpolys};
const int numparams={numparams};
float[numpolys][numparams] polys;
//float A;

float calcpolys(float x){{
    //x-=A;
    float[numparams] pows;
    pows[0]=1;
    for(int j=1;j<numparams;j++){{
        pows[j]=pows[j-1]*x;
    }}

    float s=0;
    for(int i=0;i<numpolys;i++){{
        float[numparams] poly=polys[i];
        float d=0;
        for (int j = 0; j < numparams; j++) {{
            d += poly[j] * pows[j];
        }}
        s+=abs(d);
    }}
    return s;
}}
const float inf=pow(2,1024);

float rootpolysquartic(){{
    float minx=inf;
    vec4 roots=solveQuartic(polys[0][4],polys[0][3],polys[0][2],polys[0][1],polys[0][0]);
    for(int i=0;i<5;i++){{
    float root=roots[i];
        if((!isnan(root)) && root>0){{
            minx=min(minx,root);
        }}
    }}
    return minx;
}}

//void compilepolys(vec3 p,vec3 d,float a){{
void compilepolys(vec3 p,vec3 d){{
float a=0;
float ox=p.x;
float oy=p.y;
float oz=p.z;
float dx=d.x;
float dy=d.y;
float dz=d.z;
{plangsgl}
{polyset}
}}


"""

    return fundec

def stringifynodespy(endpoint):
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


ox,oy,oz,dx,dy,dz,a=[opg.VarNode(name) for name in "ox,oy,oz,dx,dy,dz,a".split(",")]
x,y,z=ox+a*dx,oy+a*dy,oz+a*dz#defines a ray in 3d o+a*d where o is the start and d is the direction


endpoint=opg.EndpointNode(funtovisualize(x,y,z))
simplify(endpoint)
#print(endpoint.asplanstr(compact=True).replace("node","n"))


derivatives=[endpoint.parents]
for i in range(1,5):
    #print( type(derivatives[-1][0]))
    
    derivatives.append([p.backpropergation([a])[0]/i for p in derivatives[-1]])
# for i in enumerate(derivatives):
#     f=math.factorial(i)
#     d=derivatives[i]
#     for j in range(len(d)):
#         d[j]=d[j]/f

endpoint=opg.EndpointNode([opg.EndpointNode(d) for d in derivatives])
simplify(endpoint)

#longendpoint=opg.EndpointNode([deriv for poly in zip(*[p.parents for p in endpoint.parents])for deriv in poly])
longendpoint=opg.EndpointNode([params for deriv in endpoint.parents for params in deriv.parents])
#print(stringifynodesglsl(longendpoint))
fundec=stringifynodesglsl(longendpoint)



with open("./spheretracing2/scene3.glsl","w")as f:
    f.write(fundec)
with open("./spheretracing2/scene3.py","w")as f:
    f.write(stringifynodespy(longendpoint))
#print(fundec)
#print(longendpoint.asplanstr(compact=True).replace("node","n"))
print(stringifynodespy(longendpoint))