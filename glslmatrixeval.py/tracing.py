import sys
sys.path.append('./glslcompile')
import tracing_helper
import numpy as np
import glslprog

sys.path.append('./')
from variable_listener import Server
import algebra.dcga as dcga
import matrixalgebra3
server=Server()



import sys
    sys.path.append('./')
    import algebra.dcga as dcga
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


#scene=opgraphtofuncasadi.generate_glsl_code(funtovisualize)
prog=glslprog.glslprogramm(version="440")
prog.parts.append(glslprog.glslprogrammpart(bodypath="./glslcompile/fragmentshader6.glsl"))
prog.parts.append(scene)


fragment_src = str(prog)
#print(fragment_src)
with open("./glslcompile/lastfragment.glsl","w")as f:
    f.write(fragment_src)


window=tracing_helper.mywindow(800,600,fragment_src)
tracing_helper.keymouseeventhandler(window)
window.loop()