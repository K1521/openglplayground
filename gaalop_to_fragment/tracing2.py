import sys
sys.path.append('./glslcompile')
import tracing_helper
import numpy as np
import glslprog

sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
from matrixalgebra4 import TraceBasisDCGA
server=Server()




tracer=TraceBasisDCGA()

def funtoviz(point):
    obj=dcga.toroid(2,0.5)
    #obj=dcga.Plane(1,1,1,0.01)
    obj=dcga.Plane(1,1,1,0.01).outer(dcga.toroid(2,0.5))
    #obj=dcga.Plane(1,1,1,0.01).inner(dcga.toroid(2,0.5))

    #obj=sanwich(t,obj)
    prod=point.inner(obj)
    return prod
funmat=tracer.trace_linear(funtoviz)


print(funmat)
print(tracer.poly_basis_vec)

#print(tracer.poly_basis_monoms)
#scene=opgraphtofuncasadi.generate_glsl_code(funtovisualize)
prog=glslprog.glslprogramm(version="440")
prog.parts.append(glslprog.glslprogrammpart(bodypath="./gaalop_to_fragment/fragmentshader10.glsl"))
prog.parts.append(glslprog.glslprogrammpart(header=f"""
const int polybasislength={len(tracer.poly_basis_vec)};
const int numpolys={len(funmat)};
//const int MAXPOLYDEGREE=4;
"""))
print(prog.parts[-1].header)

fragment_src = str(prog)
#print(fragment_src)
with open("./gaalop_to_fragment/lastfragment.glsl","w")as f:
    f.write(fragment_src)


window=tracing_helper.mywindow(800,600,fragment_src)
#window.variables.additem(key,value,"glUniform1fv")
#window.variables.additem(key,value,"glUniform1iv")
tracing_helper.keymouseeventhandler(window)
mat=(funmat)
print(mat)
window.variables.additem("coefficientsxyz[0][0]",funmat,"glUniform1fv")






import cProfile
import pstats

profiler=cProfile.Profile()
with profiler:
    for i in range(10):
        window.loopiter()
        window.variables["cameraPos"]+=np.array([0,0.03,0])

pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(10)



#polypowmat=tracer.polypowmat()

window.loop()