import sys
sys.path.append('./glslcompile')
import tracing_helper
import numpy as np
import glslprog

sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
from matrixalgebra3 import TraceBasis
server=Server()




tracer=TraceBasis(dcga.point)

def funtoviz(point):
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,1)
    #obj=dcga.Plane(1,1,1,0.01).outer(dcga.toroid(2,0.5))

    #obj=sanwich(t,obj)
    prod=point.inner(obj)
    return prod
funmat=tracer.trace_linear(funtoviz)
shiftmat=tracer.move_basis(1,0,0)@tracer.move_basis(0,1,0)
print(funmat)
print(tracer.point_mat)
print(str((funmat@tracer.point_mat@shiftmat@tracer.poly_basis_vec)[0]).replace("**","^"))

print(funmat)

#print(tracer.poly_basis_monoms)
#scene=opgraphtofuncasadi.generate_glsl_code(funtovisualize)
prog=glslprog.glslprogramm(version="440")
prog.parts.append(glslprog.glslprogrammpart(bodypath="./glslmatrixeval/fragmentshader6.glsl"))
prog.parts.append(glslprog.glslprogrammpart(header=f"""
const int polybasislength={len(tracer.poly_basis_monoms)};
ivec3[polybasislength] polybasis={{{f",".join(f"{{{monom.x},{monom.y},{monom.z}}}"for monom in tracer.poly_basis_monoms)}}};//const is slower
const int numpolys={len(funmat)};
const int MAXPOLYDEGREE={max(monom.degree() for monom in tracer.poly_basis_monoms)};
"""))
print(prog.parts[-1].header)

fragment_src = str(prog)
#print(fragment_src)
with open("./glslmatrixeval/lastfragment.glsl","w")as f:
    f.write(fragment_src)


window=tracing_helper.mywindow(800,600,fragment_src)
#window.variables.additem(key,value,"glUniform1fv")
#window.variables.additem(key,value,"glUniform1iv")
tracing_helper.keymouseeventhandler(window)
mat=(funmat@tracer.point_mat@tracer.move_basis(0,0,0)).ravel()
print(mat)
window.variables.additem("coefficientsxyz[0][0]",mat,"glUniform1fv")

class updatemat:
    def __init__(self):
        self.lastupdate=-1
        #self.tracer=tracer
        self.m1=funmat@tracer.point_mat

    def __call__(self):
        campos=window.variables.variables["cameraPos"]
        update=campos.lastupdate
        if update!=self.lastupdate:
            self.lastupdate=update
            mat=(self.m1@tracer.move_basis(*(campos.value)))
            window.variables["coefficientsxyz[0][0]"]=mat
window.loopcallbacks.append(updatemat())





import cProfile,pstats
import pstats

profiler=cProfile.Profile()
with profiler:
    for i in range(300):
        window.loopiter()
        window.variables["cameraPos"]+=np.array([0,0.03,0])

pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(10)
window.loop()
pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(7)