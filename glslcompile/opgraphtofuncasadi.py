
import sys
sys.path.append('./')
import algebra.dcga as dcga
import casadi as ca
#import math
from glslprog import glslprogrammpart
from types import SimpleNamespace




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
    header=f"""
const int numparams={numparams};
const int numpolys={numpolys};
float[numpolys][numparams] polys;
float calcpolys(float x);
float rootpolysquartic();
void compilepolys(vec3 p,vec3 d);

"""
    body=f"""

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

    return glslprogrammpart(header,body,bodyprio=-10,headerprio=10)




def make_casadi_endpoint(f):
    """
    Translates a symbolic expression into a CasADi endpoint-like symbolic computation.
    Replaces OPG-like graph handling with CasADi symbolic operations.
    """
    fun=f
    f=lambda x,y,z:ca.vertcat(*fun(x,y,z))
    # Define symbolic variables
    #ox, oy, oz, dx, dy, dz, a = ca.SX.sym("ox"), ca.SX.sym("oy"), ca.SX.sym("oz"), ca.SX.sym("dx"), ca.SX.sym("dy"), ca.SX.sym("dz"), ca.SX.sym("a")
    ox, oy, oz, dx, dy, dz, a=[ca.SX.sym(name) for name in "ox,oy,oz,dx,dy,dz,a".split(",")]
    # Define the parametric ray equations
    x = ox + a * dx
    y = oy + a * dy
    z = oz + a * dz

    # Compute symbolic function f(x, y, z) expression
    endpoint_expr = f(x, y, z)  # Replace this with the symbolic expression you compute

    # Compute derivatives
    derivatives = [endpoint_expr]  # Initial endpoint computation
    for i in range(1, 5):
        # Compute nth derivatives
        derivative_expr = ca.jacobian(derivatives[-1], a)  # nth symbolic derivative
        derivatives.append(derivative_expr)

    # Return symbolic derivatives encapsulated in a namespace-like structure
    return derivatives


import re

def parse_and_generate_glsl(raw_output):
    """
    Parse raw CasADi symbolic output and generate GLSL code.
    Args:
    - raw_output: A string containing raw symbolic expressions with temporary variables (@n).

    Returns:
    - GLSL-compatible string with variables and expressions.
    """
    # Extract lines with assignments (e.g., @1=...)
    assignments = re.findall(r"@(\d+)=([^,;]+)", raw_output)

    # Map CasADi temporary variables to GLSL-compatible names
    variable_map = {f"@{var}": f"n{var}" for var, _ in assignments}

    # Replace temporary variables (@n) with GLSL-style names (n0, n1, ...)
    glsl_code_lines = []
    for var, expr in assignments:
        # Replace other temporary variables within the expression
        for casadi_var, glsl_var in variable_map.items():
            expr = expr.replace(casadi_var, glsl_var)
        glsl_code_lines.append(f"float n{var} = {expr.strip()};")

    # Add a final placeholder for Taylor coefficients
    taylor_lines = [f"polys[0][{i}] = n{i};" for i in range(len(assignments))]

    # Combine into final GLSL code
    glsl_code = "\n".join(glsl_code_lines + taylor_lines)
    return glsl_code


# Example: Raw CasADi symbolic output (truncated for brevity)





def generate_glsl_code(f):
    derivatives=make_casadi_endpoint(f)
    """
    Translates symbolic computed Taylor series coefficients into GLSL-like shader code.
    Args:
    - taylor_series: The computed symbolic derivatives (and their substitutions).

    Returns:
    - GLSL function string to compute the Taylor expansion using computed coefficients.
    """
    # Extract the computed derivatives
    cse_lines = []
    for idx, expr in enumerate(derivatives):
        # Convert each symbolic expression to GLSL-compatible string
        expr_glsl = str(expr)
        cse_lines.append(f"float n{idx} = {expr_glsl};")  # Create GLSL-like variable assignments.

    # Write the final GLSL function string
    header = """
    const int numparams = 5;
    const int numpolys = 1;
    float polys[numpolys][numparams];
    """
    glsl_code="""
    void compilepolys(vec3 p, vec3 d) {
        float ox = p.x;
        float oy = p.y;
        float oz = p.z;
        float dx = d.x;
        float dy = d.y;
        float dz = d.z;
    """
    # Add computed Taylor series derivatives
    glsl_code += "\n".join(cse_lines)
    glsl_code += "\n"
    
    # Map computed derivatives to GLSL computations
    body_lines = [f"polys[0][{idx}] = n{idx};" for idx in range(len(derivatives))]

    glsl_code += "\n".join(body_lines)
    glsl_code += """
    }
    """
    return glslprogrammpart(header,glsl_code,bodyprio=-10,headerprio=10)